"""추론 백엔드 3종을 같은 인터페이스로 감싼다: TensorRT / onnxruntime / PyTorch.

    runner = make_runner("trt", engine="model_fp16.engine")
    disp = runner(left, right)        # numpy in -> numpy out [1,1,H,W]

같은 전처리·같은 지표 코드로 세 백엔드를 비교하려고 만들었다. Jetson 에서는 trt,
데스크탑에서 파이프라인을 검증할 때는 onnx, 기준값이 필요할 때는 torch 를 쓴다.

**torch 는 선택 의존성이다.** trt / onnx 백엔드는 numpy 만으로 돌아가므로 torch 가
없는 Jetson 에서도 쓸 수 있다. torch 는 `--backend torch` 를 쓸 때만 import 한다.

TensorRT 백엔드의 GPU 메모리는 다음 순서로 잡는다:
    cuda-python (from cuda import cudart)  ->  pycuda
JetPack 에는 보통 cuda-python 이 들어 있고, 없으면 `pip install cuda-python` 하나면
된다. TRT 8.5 이전의 binding API 와 8.5+ 의 tensor API 를 둘 다 지원한다.
"""

import os

import numpy as np


class TorchRunner:
    """PyTorch 기준값. FP16/BF16 열화를 재려면 비교 대상이 있어야 한다.

    이 클래스를 만들 때만 torch 를 import 한다 (Jetson 에는 torch 가 없을 수 있다).
    """

    def __init__(self, model, device="cuda:0", precision="fp32"):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.precision = precision
        if precision == "fp16":
            self.model = self.model.half()
        elif precision == "bf16":
            self.model = self.model.to(torch.bfloat16)
        self.name = "torch-{}".format(precision)

    def __call__(self, left, right):
        torch = self.torch
        dt = {"fp32": torch.float32, "fp16": torch.float16,
              "bf16": torch.bfloat16}[self.precision]
        with torch.no_grad():
            out = self.model({"left": torch.from_numpy(left).to(self.device, dt),
                              "right": torch.from_numpy(right).to(self.device, dt)})
            if isinstance(out, dict):
                out = out["disp_pred"]
            if out.dim() == 3:
                out = out.unsqueeze(1)
            return out.float().cpu().numpy()


class ORTRunner:
    """onnxruntime. 데스크탑에서 전처리·지표 코드를 검증하는 용도."""

    def __init__(self, onnx_path, device="cuda:0", threads=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            # Slurm cgroup 아래에서는 스레드 수를 명시하지 않으면 onnxruntime 이
            # 없는 코어에 affinity 를 걸려다 pthread_setaffinity_np 오류를 쏟는다.
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(onnx_path, opts, providers=providers)
        self.name = "onnxruntime({})".format(providers[0].replace("ExecutionProvider", ""))

    def __call__(self, left, right):
        return self.sess.run(None, {"left": left, "right": right})[0].astype(np.float32)


class _CudaMem:
    """GPU 메모리 할당/복사. cuda-python 을 먼저, 없으면 pycuda 를 쓴다.

    torch 는 일부러 쓰지 않는다 - 이 백엔드의 존재 이유가 torch 없는 Jetson 이다.
    """

    def __init__(self):
        try:
            from cuda import cudart          # cuda-python (JetPack 에 보통 포함)
            self.cudart = cudart
            self.kind = "cuda-python"
            return
        except ImportError:
            pass
        try:
            import pycuda.autoinit           # noqa: F401  (컨텍스트 초기화 부작용)
            import pycuda.driver as cuda
            self.cuda = cuda
            self.kind = "pycuda"
            return
        except ImportError:
            pass
        raise SystemExit(
            "GPU 메모리를 잡을 방법이 없다. 둘 중 하나를 설치할 것:\n"
            "  pip install cuda-python   (권장. NVIDIA 공식, Jetson 에서 가장 가볍다)\n"
            "  pip install pycuda")

    def _check(self, err):
        if int(err) != 0:
            raise RuntimeError("CUDA 오류 {}".format(err))

    def alloc(self, nbytes):
        if self.kind == "cuda-python":
            err, ptr = self.cudart.cudaMalloc(nbytes)
            self._check(err)
            return ptr
        return self.cuda.mem_alloc(nbytes)

    def free(self, ptr):
        if self.kind == "cuda-python":
            self._check(self.cudart.cudaFree(ptr))
        else:
            ptr.free()

    def htod(self, dst, arr):
        if self.kind == "cuda-python":
            self._check(self.cudart.cudaMemcpy(
                dst, arr.ctypes.data, arr.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)[0])
        else:
            self.cuda.memcpy_htod(dst, arr)

    def dtoh(self, arr, src):
        if self.kind == "cuda-python":
            self._check(self.cudart.cudaMemcpy(
                arr.ctypes.data, src, arr.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)[0])
        else:
            self.cuda.memcpy_dtoh(arr, src)

    def ptr(self, p):
        return int(p)

    def sync(self):
        if self.kind == "cuda-python":
            self._check(self.cudart.cudaDeviceSynchronize())
        else:
            self.cuda.Context.synchronize()


class TRTRunner:
    """TensorRT 엔진. numpy 로 주고받고, torch 없이도 돈다."""

    _NP = {"float32": np.float32, "float16": np.float16,
           "int32": np.int32, "int8": np.int8}

    def __init__(self, engine_path, device="cuda:0"):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("엔진 로드 실패: {} (TRT 버전/GPU 가 빌드 때와 다른가?)"
                               .format(engine_path))
        self.ctx = self.engine.create_execution_context()
        self.mem = _CudaMem()

        # TRT 8.5+ 는 tensor API, 그 이전은 binding API
        self.tensor_api = hasattr(self.engine, "num_io_tensors")
        self.name = "tensorrt {} ({})".format(trt.__version__, self.mem.kind)

        if self.tensor_api:
            self.io_names = [self.engine.get_tensor_name(i)
                             for i in range(self.engine.num_io_tensors)]
            self.inputs = [n for n in self.io_names
                           if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        else:
            self.io_names = [self.engine.get_binding_name(i)
                             for i in range(self.engine.num_bindings)]
            self.inputs = [n for n in self.io_names
                           if self.engine.binding_is_input(self.engine.get_binding_index(n))]
        self.outputs = [n for n in self.io_names if n not in self.inputs]
        self._dev = {}          # name -> device ptr
        self._host_out = {}     # name -> numpy 버퍼

    def _np_dtype(self, name):
        if self.tensor_api:
            dt = self.engine.get_tensor_dtype(name)
        else:
            dt = self.engine.get_binding_dtype(self.engine.get_binding_index(name))
        return self._NP.get(str(dt).split(".")[-1].lower(), np.float32)

    def _shape(self, name):
        if self.tensor_api:
            return tuple(self.ctx.get_tensor_shape(name))
        return tuple(self.ctx.get_binding_shape(self.engine.get_binding_index(name)))

    def __call__(self, left, right):
        feed = {"left": left, "right": right}
        if not set(self.inputs) <= set(feed):     # 이름이 다르면 순서로 받는다
            feed = dict(zip(self.inputs, [left, right]))

        for name in self.inputs:
            arr = np.ascontiguousarray(feed[name], dtype=self._np_dtype(name))
            if self.tensor_api:
                self.ctx.set_input_shape(name, tuple(arr.shape))
            if name not in self._dev:
                self._dev[name] = self.mem.alloc(arr.nbytes)
            self.mem.htod(self._dev[name], arr)
            if self.tensor_api:
                self.ctx.set_tensor_address(name, self.mem.ptr(self._dev[name]))

        for name in self.outputs:
            shape = self._shape(name)
            host = self._host_out.get(name)
            if host is None or host.shape != shape:
                host = np.empty(shape, dtype=self._np_dtype(name))
                self._host_out[name] = host
                if name in self._dev:
                    self.mem.free(self._dev.pop(name))
            if name not in self._dev:
                self._dev[name] = self.mem.alloc(host.nbytes)
            if self.tensor_api:
                self.ctx.set_tensor_address(name, self.mem.ptr(self._dev[name]))

        if self.tensor_api:
            ok = self.ctx.execute_async_v3(0)
        else:
            ok = self.ctx.execute_v2([self.mem.ptr(self._dev[n]) for n in self.io_names])
        if not ok:
            raise RuntimeError("TensorRT 실행 실패")
        self.mem.sync()

        name = self.outputs[0]
        self.mem.dtoh(self._host_out[name], self._dev[name])
        out = self._host_out[name].astype(np.float32)
        return out if out.ndim == 4 else out[None]


def make_runner(backend, engine=None, onnx=None, model=None,
                device="cuda:0", precision="fp32", threads=None):
    if backend == "trt":
        if not engine or not os.path.isfile(engine):
            raise SystemExit("--engine 이 필요하다 (TensorRT 엔진 경로)")
        return TRTRunner(engine, device)
    if backend == "onnx":
        if not onnx or not os.path.isfile(onnx):
            raise SystemExit("--onnx 가 필요하다")
        return ORTRunner(onnx, device, threads)
    if backend == "torch":
        if model is None:
            raise SystemExit("torch 백엔드는 모델 객체가 필요하다 (--cfg/--ckpt)")
        return TorchRunner(model, device, precision)
    raise SystemExit("모르는 backend: {}".format(backend))
