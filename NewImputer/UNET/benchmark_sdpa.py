"""
v6 SDPA 消融: 对比 FlashAttention (SDPA) 开/关的速度和精度
用法: python benchmark_sdpa.py --device 1 --checkpoint <path>
"""
import torch
import time
import argparse
import sys
sys.path.insert(0, '/home/duanlei/PriSTI/NewImputer/UNET')

from aqi36_lightning_datamodule import AQI36_DataModule
from transformer_imputer_v6 import transformer_imputer


def benchmark_inference(model, dataloader, device, use_sdpa=True, warmup=3, runs=3):
    """测量推理速度和结果"""
    model.eval()
    
    # 收集所有 batch
    batches = list(dataloader)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            batch = batches[0]
            coeffs = batch["coeffs"].to(device).float()
            cond_mask = batch["cond_mask"].to(device).float()
            if use_sdpa:
                model(coeffs, cond_mask)
            else:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                ):
                    model(coeffs, cond_mask)
    
    torch.cuda.synchronize()
    
    # 计时
    times = []
    for run in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            for batch in batches:
                coeffs = batch["coeffs"].to(device).float()
                cond_mask = batch["cond_mask"].to(device).float()
                if use_sdpa:
                    out, _ = model(coeffs, cond_mask)
                else:
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=False, enable_math=True, enable_mem_efficient=False
                    ):
                        out, _ = model(coeffs, cond_mask)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return times


def benchmark_training_step(model, dataloader, device, use_sdpa=True, steps=20):
    """测量训练速度"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    batches = list(dataloader)
    
    # Warmup
    for i in range(3):
        batch = batches[i % len(batches)]
        coeffs = batch["coeffs"].to(device).float()
        cond_mask = batch["cond_mask"].to(device).float()
        observed = batch["observed_data"].to(device).float().permute(0, 2, 1)
        eval_mask = (batch["observed_mask"].to(device).float() - batch["cond_mask"].to(device).float()).permute(0, 2, 1)
        
        optimizer.zero_grad()
        if use_sdpa:
            out, _ = model(coeffs, cond_mask)
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                out, _ = model(coeffs, cond_mask)
        loss = (torch.abs(out.permute(0, 2, 1) - observed) * eval_mask).sum() / eval_mask.sum()
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    
    # 正式计时
    start = time.perf_counter()
    for i in range(steps):
        batch = batches[i % len(batches)]
        coeffs = batch["coeffs"].to(device).float()
        cond_mask = batch["cond_mask"].to(device).float()
        observed = batch["observed_data"].to(device).float().permute(0, 2, 1)
        eval_mask = (batch["observed_mask"].to(device).float() - batch["cond_mask"].to(device).float()).permute(0, 2, 1)
        
        optimizer.zero_grad()
        if use_sdpa:
            out, _ = model(coeffs, cond_mask)
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                out, _ = model(coeffs, cond_mask)
        loss = (torch.abs(out.permute(0, 2, 1) - observed) * eval_mask).sum() / eval_mask.sum()
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    
    device = f"cuda:{args.device}"
    
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(args.device)}")
    print()
    
    # 加载数据 (用测试集就够了)
    dm = AQI36_DataModule(device=device, preimpute_flag="Forward", freq_flag=False)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    
    print(f"Test batches: {len(test_loader)}")
    
    for mode_name, use_sdpa in [("SDPA ON (FlashAttention)", True), ("SDPA OFF (Math kernel)", False)]:
        print(f"\n{'='*50}")
        print(f"  {mode_name}")
        print(f"{'='*50}")
        
        # 创建新模型 (每次从零开始，保证公平)
        model = transformer_imputer(device=device, exit_threshold=0.0)
        
        # 推理速度
        times = benchmark_inference(model, test_loader, device, use_sdpa=use_sdpa)
        avg_time = sum(times) / len(times)
        print(f"  Inference: {avg_time:.3f}s ({len(test_loader)} batches, avg of {len(times)} runs)")
        for i, t in enumerate(times):
            print(f"    Run {i}: {t:.3f}s")
        
        # 训练速度
        train_loader = dm.train_dataloader()
        elapsed, steps = benchmark_training_step(model, train_loader, device, use_sdpa=use_sdpa, steps=20)
        print(f"  Training: {elapsed:.3f}s ({steps} steps, {elapsed/steps*1000:.1f}ms/step)")
        
        # 清理
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*50}")
    print("Done!")
