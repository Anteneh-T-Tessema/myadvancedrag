"""
Hardware Inspector
Detects CPU, GPU (Apple Silicon / NVIDIA / AMD), RAM, disk, and
recommends the optimal embedding + LLM models for this machine.
"""

import platform
import subprocess
import shutil
import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


# ── Model recommendations keyed by hardware tier ──────────────────────────────

EMBED_RECOMMENDATIONS = {
    "apple_silicon_high":  ["nomic-embed-text", "BAAI/bge-m3", "all-MiniLM-L6-v2"],
    "apple_silicon_low":   ["all-MiniLM-L6-v2", "nomic-embed-text"],
    "nvidia_high":         ["BAAI/bge-m3", "nomic-embed-text", "all-MiniLM-L6-v2"],
    "nvidia_low":          ["all-MiniLM-L6-v2", "nomic-embed-text"],
    "cpu_only":            ["all-MiniLM-L6-v2"],
}

LLM_RECOMMENDATIONS = {
    "apple_silicon_high":  ["llama3.1:8b", "llama3.2:3b", "mistral:7b", "gemma2:9b"],
    "apple_silicon_low":   ["llama3.2:1b", "llama3.2:3b", "phi3:mini"],
    "nvidia_high":         ["llama3.1:8b", "mistral:7b", "gemma2:9b", "llama3.2:3b"],
    "nvidia_low":          ["llama3.2:3b", "phi3:mini", "llama3.2:1b"],
    "cpu_only":            ["llama3.2:1b", "phi3:mini"],
}

POPULAR_OLLAMA_MODELS = [
    {"name": "llama3.2:1b",   "size_gb": 0.8,  "desc": "Meta Llama 3.2 1B — ultra-fast, low RAM"},
    {"name": "llama3.2:3b",   "size_gb": 2.0,  "desc": "Meta Llama 3.2 3B — fast, good quality"},
    {"name": "llama3.2",      "size_gb": 2.0,  "desc": "Meta Llama 3.2 (latest)"},
    {"name": "llama3.1:8b",   "size_gb": 4.7,  "desc": "Meta Llama 3.1 8B — best open-source quality"},
    {"name": "mistral:7b",    "size_gb": 4.1,  "desc": "Mistral 7B — strong reasoning"},
    {"name": "gemma2:9b",     "size_gb": 5.4,  "desc": "Google Gemma 2 9B — excellent instruction following"},
    {"name": "phi3:mini",     "size_gb": 2.3,  "desc": "Microsoft Phi-3 Mini — efficient & capable"},
    {"name": "codellama:7b",  "size_gb": 3.8,  "desc": "Meta Code Llama 7B — code specialised"},
    {"name": "nomic-embed-text", "size_gb": 0.3, "desc": "Nomic Embed — 8192 ctx embedding model"},
    {"name": "mxbai-embed-large","size_gb": 0.7, "desc": "MixedBread Embed Large — high accuracy"},
    {"name": "llama-guard3:8b",  "size_gb": 4.7, "desc": "Meta Llama Guard 3 — safety classifier"},
]


@dataclass
class CPUInfo:
    brand: str
    architecture: str
    physical_cores: int
    logical_cores: int
    frequency_mhz: float
    is_apple_silicon: bool


@dataclass
class GPUInfo:
    name: str
    vendor: str          # apple | nvidia | amd | unknown
    vram_gb: float
    metal_support: bool  # Apple Metal
    cuda_support: bool   # NVIDIA CUDA


@dataclass
class MemoryInfo:
    total_gb: float
    available_gb: float
    used_percent: float


@dataclass
class DiskInfo:
    total_gb: float
    free_gb: float
    used_percent: float


@dataclass
class HardwareProfile:
    system: str
    node: str
    os_version: str
    python_version: str
    cpu: CPUInfo
    gpus: List[GPUInfo]
    memory: MemoryInfo
    disk: DiskInfo
    ollama_installed: bool
    ollama_version: str
    hardware_tier: str          # apple_silicon_high/low | nvidia_high/low | cpu_only
    recommended_embed_models: List[str] = field(default_factory=list)
    recommended_llm_models: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recommended_embed_models"] = self.recommended_embed_models
        d["recommended_llm_models"] = self.recommended_llm_models
        d["warnings"] = self.warnings
        return d


# ── Detector ──────────────────────────────────────────────────────────────────

class HardwareInspector:
    """Cross-platform hardware detector optimised for macOS Apple Silicon and
    NVIDIA Linux/Windows development machines."""

    def inspect(self) -> HardwareProfile:
        cpu    = self._detect_cpu()
        gpus   = self._detect_gpus()
        mem    = self._detect_memory()
        disk   = self._detect_disk()
        ollama_installed, ollama_ver = self._detect_ollama()
        tier   = self._compute_tier(cpu, gpus, mem)
        warns  = self._compute_warnings(mem, disk, ollama_installed)

        return HardwareProfile(
            system=platform.system(),
            node=platform.node(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            cpu=cpu,
            gpus=gpus,
            memory=mem,
            disk=disk,
            ollama_installed=ollama_installed,
            ollama_version=ollama_ver,
            hardware_tier=tier,
            recommended_embed_models=EMBED_RECOMMENDATIONS.get(tier, ["all-MiniLM-L6-v2"]),
            recommended_llm_models=LLM_RECOMMENDATIONS.get(tier, ["llama3.2:1b"]),
            warnings=warns,
        )

    # ── CPU ───────────────────────────────────────────────────────────────────

    def _detect_cpu(self) -> CPUInfo:
        brand = platform.processor() or "Unknown"
        arch  = platform.machine()
        is_apple = self._is_apple_silicon()

        try:
            import psutil
            try:
                freq = psutil.cpu_freq()
                mhz = freq.current if freq else 0.0
            except (FileNotFoundError, AttributeError, PermissionError):
                mhz = 0.0
            phys = psutil.cpu_count(logical=False) or 1
            logi = psutil.cpu_count(logical=True)  or 1
        except ImportError:
            phys = os.cpu_count() or 1
            logi = phys
            mhz  = 0.0

        # Better brand on macOS Apple Silicon
        if is_apple and platform.system() == "Darwin":
            raw = self._run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if raw:
                brand = raw.strip()

        return CPUInfo(
            brand=brand,
            architecture=arch,
            physical_cores=phys,
            logical_cores=logi,
            frequency_mhz=mhz,
            is_apple_silicon=is_apple,
        )

    def _is_apple_silicon(self) -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    # ── GPU ───────────────────────────────────────────────────────────────────

    def _detect_gpus(self) -> List[GPUInfo]:
        gpus = []

        if platform.system() == "Darwin":
            gpus.extend(self._detect_apple_gpu())
        else:
            gpus.extend(self._detect_nvidia_gpu())
            gpus.extend(self._detect_amd_gpu())

        return gpus or [GPUInfo(name="Unknown/Integrated", vendor="unknown",
                                vram_gb=0.0, metal_support=False, cuda_support=False)]

    def _detect_apple_gpu(self) -> List[GPUInfo]:
        gpus = []
        raw = self._run(["system_profiler", "SPDisplaysDataType", "-json"])
        if not raw:
            return gpus
        try:
            data = json.loads(raw)
            displays = data.get("SPDisplaysDataType", [])
            for disp in displays:
                name  = disp.get("sppci_model", "Apple GPU")
                vram_str = disp.get("sppci_vram", "0 MB")
                vram_gb = self._parse_vram(vram_str)
                gpus.append(GPUInfo(
                    name=name,
                    vendor="apple",
                    vram_gb=vram_gb,
                    metal_support=True,
                    cuda_support=False,
                ))
        except Exception:
            pass
        return gpus

    def _detect_nvidia_gpu(self) -> List[GPUInfo]:
        gpus = []
        raw = self._run([
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        if not raw:
            return gpus
        for line in raw.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    vram_gb = round(int(parts[1]) / 1024, 1)
                except ValueError:
                    vram_gb = 0.0
                gpus.append(GPUInfo(
                    name=parts[0],
                    vendor="nvidia",
                    vram_gb=vram_gb,
                    metal_support=False,
                    cuda_support=True,
                ))
        return gpus

    def _detect_amd_gpu(self) -> List[GPUInfo]:
        raw = self._run(["rocm-smi", "--showproductname"])
        if not raw:
            return []
        return [GPUInfo(
            name=raw.strip().split("\n")[0] if raw else "AMD GPU",
            vendor="amd",
            vram_gb=0.0,
            metal_support=False,
            cuda_support=False,
        )]

    # ── Memory ────────────────────────────────────────────────────────────────

    def _detect_memory(self) -> MemoryInfo:
        try:
            import psutil
            vm = psutil.virtual_memory()
            return MemoryInfo(
                total_gb=round(vm.total / 1e9, 1),
                available_gb=round(vm.available / 1e9, 1),
                used_percent=vm.percent,
            )
        except ImportError:
            # macOS fallback via sysctl
            raw = self._run(["sysctl", "-n", "hw.memsize"])
            total = int(raw.strip()) / 1e9 if raw else 0.0
            return MemoryInfo(total_gb=round(total, 1), available_gb=0.0, used_percent=0.0)

    # ── Disk ──────────────────────────────────────────────────────────────────

    def _detect_disk(self) -> DiskInfo:
        try:
            import psutil
            usage = psutil.disk_usage("/")
            return DiskInfo(
                total_gb=round(usage.total / 1e9, 1),
                free_gb=round(usage.free / 1e9, 1),
                used_percent=usage.percent,
            )
        except ImportError:
            st = os.statvfs("/")
            total = st.f_blocks * st.f_frsize / 1e9
            free  = st.f_bavail * st.f_frsize / 1e9
            pct   = round((1 - free / total) * 100, 1) if total else 0.0
            return DiskInfo(total_gb=round(total, 1), free_gb=round(free, 1), used_percent=pct)

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _detect_ollama(self):
        path = shutil.which("ollama")
        if not path:
            return False, ""
        ver = self._run(["ollama", "--version"]) or ""
        return True, ver.strip()

    # ── Tier ──────────────────────────────────────────────────────────────────

    def _compute_tier(self, cpu: CPUInfo, gpus: List[GPUInfo], mem: MemoryInfo) -> str:
        ram = mem.total_gb

        if cpu.is_apple_silicon:
            # Apple Silicon: unified memory — RAM IS VRAM
            return "apple_silicon_high" if ram >= 16 else "apple_silicon_low"

        has_nvidia = any(g.vendor == "nvidia" for g in gpus)
        if has_nvidia:
            max_vram = max((g.vram_gb for g in gpus if g.vendor == "nvidia"), default=0)
            return "nvidia_high" if max_vram >= 8 else "nvidia_low"

        return "cpu_only"

    # ── Warnings ─────────────────────────────────────────────────────────────

    def _compute_warnings(self, mem: MemoryInfo, disk: DiskInfo, ollama_ok: bool) -> List[str]:
        warns = []
        if mem.total_gb < 8:
            warns.append("⚠️  Less than 8 GB RAM — stick to 1B parameter models.")
        if disk.free_gb < 10:
            warns.append(f"⚠️  Only {disk.free_gb:.1f} GB free disk — models need 1–6 GB each.")
        if not ollama_ok:
            warns.append("⚠️  Ollama not found. Install from https://ollama.ai")
        return warns

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run(self, cmd: List[str]) -> Optional[str]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            return result.stdout if result.returncode == 0 else None
        except Exception:
            return None

    @staticmethod
    def _parse_vram(vram_str: str) -> float:
        """Parse '16 GB', '8192 MB', etc. → float GB."""
        parts = vram_str.strip().split()
        if len(parts) < 2:
            return 0.0
        try:
            val = float(parts[0].replace(",", ""))
            unit = parts[1].upper()
            if unit == "MB":
                return round(val / 1024, 1)
            return round(val, 1)
        except ValueError:
            return 0.0
