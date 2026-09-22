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


def cuda_mem_probe():
    """CUDA 컨텍스트를 먼저 띄우고 free/total 을 재는 도구.

    torch.cuda.max_memory_allocated() 에 대응하는 값을 얻으려면 **컨텍스트 초기화
    이후**를 기준선으로 잡아야 한다. 아무 CUDA 호출 전에 재면 이후 생기는 컨텍스트
    (수백 MB)가 델타에 통째로 들어가 비교가 안 된다.

    반환: free_gb() 를 가진 객체. cuda-python 이 없으면 None.
    """
    try:
        mem = _CudaMem()
    except SystemExit:
        return None
    if mem.kind != "cuda-python":
        return None

    # cudaFree(0) 은 컨텍스트만 만들고 아무것도 해제하지 않는 관용적 초기화다.
    try:
        mem.cudart.cudaFree(0)
    except Exception:
        pass

    class Probe:
        def __init__(self, mem):
            self.mem = mem

        def free_gb(self):
            err, free, total = self.mem.cudart.cudaMemGetInfo()
            if int(err) != 0:
                return None
            return free / 1024 ** 3, total / 1024 ** 3

    return Probe(mem)


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
        # cuda-python 12.8 부터 모듈 경로가 cuda.bindings.runtime 으로 바뀌었다.
        # 구버전(cuda.cudart)도 같이 받아 준다.
        for imp in ("bindings", "legacy"):
            try:
                if imp == "bindings":
                    from cuda.bindings import runtime as cudart
                else:
                    from cuda import cudart
                self.cudart = cudart
                self.kind = "cuda-python"
                self.stream = None
                return
            except ImportError:
                continue
        try:
            import pycuda.autoinit           # noqa: F401  (컨텍스트 초기화 부작용)
            import pycuda.driver as cuda
            self.cuda = cuda
            self.kind = "pycuda"
            self.stream = None
            return
        except ImportError:
            pass
        raise SystemExit(
            "GPU 메모리를 잡을 방법이 없다. 둘 중 하나를 설치할 것:\n"
            "  pip install cuda-python   (권장. NVIDIA 공식, Jetson 에서 가장 가볍다)\n"
            "  pip install pycuda")

    def _check(self, ret):
        """cuda-python 은 반환이 제각각이다: cudaMalloc -> (err, ptr),
        cudaFree/cudaDeviceSynchronize -> (err,), 일부는 err 하나. 앞의 것만 본다."""
        err = ret[0] if isinstance(ret, (tuple, list)) else ret
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

    def get_stream(self):
        """전용 스트림. 기본 스트림(0)을 쓰면 TRT 가 enqueueV3 에서 추가 동기화를
        한다고 경고하고 실제로도 느려질 수 있다."""
        if self.kind == "cuda-python":
            if getattr(self, "stream", None) is None:
                ret = self.cudart.cudaStreamCreate()
                self._check(ret)
                self.stream = ret[1]
            return int(self.stream)
        if getattr(self, "stream", None) is None:
            self.stream = self.cuda.Stream()
        return int(self.stream.handle)

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
            if getattr(self, "stream", None) is not None:
                self._check(self.cudart.cudaStreamSynchronize(self.stream))
            else:
                self._check(self.cudart.cudaDeviceSynchronize())
        elif getattr(self, "stream", None) is not None:
            self.stream.synchronize()
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

    def bind(self, left, right):
        """입력을 GPU 로 올리고 출력 버퍼를 잡는다. 벤치마크는 이걸 한 번만 하고
        execute() 만 반복해서 H2D/D2H 복사를 측정에서 뺀다 (데스크탑에서 CUDA event
        로 잰 것과 같은 조건)."""
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

    def execute(self):
        """이미 bind 된 버퍼로 추론만 한다 (복사 없음)."""
        if self.tensor_api:
            ok = self.ctx.execute_async_v3(self.mem.get_stream())
        else:
            ok = self.ctx.execute_v2([self.mem.ptr(self._dev[n]) for n in self.io_names])
        if not ok:
            raise RuntimeError("TensorRT 실행 실패")
        self.mem.sync()

    def fetch(self):
        name = self.outputs[0]
        self.mem.dtoh(self._host_out[name], self._dev[name])
        out = self._host_out[name].astype(np.float32)
        return out if out.ndim == 4 else out[None]

    def __call__(self, left, right):
        self.bind(left, right)
        self.execute()
        return self.fetch()

    def engine_memory_mb(self):
        """엔진이 실제로 잡는 GPU 메모리. cudaMemGetInfo 와 달리 다른 프로세스에
        오염되지 않는다.

          activation : 중간 텐서용 작업 공간 (TRT 가 미리 계산해 둔 값)
          weights    : 가중치. 엔진 파일 크기로 근사한다
          io         : 우리가 잡은 입출력 버퍼
        """
        act = None
        for attr in ("device_memory_size_v2", "device_memory_size"):
            if hasattr(self.engine, attr):
                try:
                    act = int(getattr(self.engine, attr))
                    break
                except Exception:
                    continue
        io = 0
        for name in self.io_names:
            shape = self._shape(name)
            n = 1
            for d in shape:
                n *= int(d)
            io += n * np.dtype(self._np_dtype(name)).itemsize
        return {"activation_mb": act / 1024 ** 2 if act is not None else None,
                "io_mb": io / 1024 ** 2}

    def mem_info_gb(self):
        """(free, total) GB. peak 메모리를 free 의 최저점으로 재기 위한 것."""
        if self.mem.kind != "cuda-python":
            return None
        err, free, total = self.mem.cudart.cudaMemGetInfo()
        if int(err) != 0:
            return None
        return free / 1024 ** 3, total / 1024 ** 3


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
