
# trainer.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from aqi36_lightning_datamodule import AQI36DataModule
from aqi36_lightning_module import GuideDiffLightning
import torch
import argparse
import yaml
from pytorch_lightning import seed_everything

device_num = 1
def main():
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default=f'cuda:{device_num}', help='Device for Attack')
    parser.add_argument(
        "--targetstrategy", type=str, default="hybrid", choices=["hybrid", "random", "historical"]
    )



    args = parser.parse_args()
    print(args)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = False
    config["model"]["target_strategy"] = args.targetstrategy
    config["diffusion"]["adj_file"] = 'AQI36'
    config["seed"] = 42


    seed = 42
    print(f"Running with seed: {seed}")
    config["seed"] = seed

    ###################
    config["channels"] = 64
    config["num_steps"] = 100
    config["diffusion_embedding_dim"] = 128
    config["adj_file"] = 'AQI36'
    config["device"] = f"cuda:{device_num}"
    config["is_adp"] = True
    config["layers"] = 4
    config["side_dim"] = 128+16
    config["nheads"] = 8
    config["proj_t"] = 16
    config["is_cross_t"] = True
    config["is_cross_s"] = True
    epochs = 200
    ##################

    seed_everything(seed)
    # 1. 数据
    dm = AQI36DataModule(batch_size=16, num_workers=16)
    dm.setup()

    # 2. 模型
    decay_flag = True
    #model = GuideDiffLightning.load_from_checkpoint(checkpoint_path="/home/duanlei/PriSTI/mid_checkpoints/last-v29.ckpt",model_config=config,epochs=epochs)
    model = GuideDiffLightning(model_config=config,epochs=epochs,decay_flag=decay_flag)
    # 3. 回调
    checkpoint = ModelCheckpoint(
        dirpath="mid_checkpoints",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")


    # 4. Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[device_num],
        callbacks=[checkpoint],
        log_every_n_steps=10,
    )

    # 5. 训练
    trainer.fit(model, dm)

    # 6. 测试（可选）
    trainer.test(model, dm)

if __name__ == "__main__":
    main()