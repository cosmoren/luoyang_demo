import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from demo_luoyang import LuoyangDataLoader

# =====================================================
# Dataset wrapper
# =====================================================

class PVShortDataset(Dataset):

    def __init__(self, loader, X_tab, X_fcst, y, t, tab_mean,
        tab_std, fcst_mean, fcst_std, sat_mean, sat_std):
        self.loader = loader
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.X_fcst = torch.tensor(X_fcst, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.times = pd.to_datetime(t)
        self.tab_mean = torch.tensor(tab_mean, dtype=torch.float32)
        self.tab_std  = torch.tensor(tab_std, dtype=torch.float32)
        self.fcst_mean = torch.tensor(fcst_mean, dtype=torch.float32)
        self.fcst_std  = torch.tensor(fcst_std, dtype=torch.float32)
        self.sat_mean = torch.tensor(sat_mean, dtype=torch.float32)
        self.sat_std  = torch.tensor(sat_std, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_tab = (self.X_tab[idx] - self.tab_mean) / self.tab_std # normalize
        X_fcst = (self.X_fcst[idx] - self.fcst_mean) / self.fcst_std # normalize
        y = self.y[idx]
        t0 = self.times[idx]

        # 调用原 dataloader
        sat_seq, sat_mask = self.loader._build_satellite_sequence(t0)
        X_sat = torch.tensor(sat_seq, dtype=torch.float32)
        X_sat_mask = torch.tensor(sat_mask, dtype=torch.float32)
        X_sat = X_sat * X_sat_mask.view(-1,1,1,1)
        n_var = len(loader.satellite_vars)
        mean = self.sat_mean[:n_var].view(1,-1,1,1)
        std  = self.sat_std[:n_var].view(1,-1,1,1)
        X_sat[:,:n_var] = (X_sat[:,:n_var] - mean) / std # normalize
        X_sat_mask = torch.tensor(sat_mask, dtype=torch.float32)

        return X_tab, X_sat, X_sat_mask, X_fcst, y

# ConvLSTMCell
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3):

        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

        self.hidden_dim = hidden_dim

    def forward(self, x, h, c):

        combined = torch.cat([x, h], dim=1)

        gates = self.conv(combined)

        i,f,o,g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

# ConvLSTM
class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.cell = ConvLSTMCell(input_dim, hidden_dim)

        self.hidden_dim = hidden_dim

    def forward(self, x):

        # x (B,T,C,H,W)

        B,T,C,H,W = x.shape

        h = torch.zeros(B,self.hidden_dim,H,W,device=x.device)
        c = torch.zeros(B,self.hidden_dim,H,W,device=x.device)

        outputs = []

        for t in range(T):

            h,c = self.cell(x[:,t],h,c)

            outputs.append(h)

        outputs = torch.stack(outputs,dim=1)

        return outputs

# =====================================================
# Satellite CNN encoder
# =====================================================

class SatelliteEncoder(nn.Module):

    def __init__(self, C):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(C,16,5,stride=2,padding=2),
            nn.ReLU(),

            nn.Conv2d(16,32,3,stride=2,padding=1),
            nn.ReLU(),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU()
        )

        self.convlstm = ConvLSTM(
            input_dim=64,
            hidden_dim=64
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):

        # x (B,T,C,H,W)

        B,T,C,H,W = x.shape

        x = x.view(B*T,C,H,W)

        feat = self.cnn(x)

        _,C2,H2,W2 = feat.shape

        feat = feat.view(B,T,C2,H2,W2)

        lstm_out = self.convlstm(feat)

        last = lstm_out[:,-1]

        pooled = self.pool(last)

        return pooled.view(B,-1)

class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim=128, patch_size=10):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B*T, C, H, W)
        x = self.proj(x)                  # (B*T, D, Hp, Wp)
        B, D, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2) # (B*T, N, D)
        return x, Hp, Wp
    
class ViTSatelliteEncoder(nn.Module):
    def __init__(
        self,
        in_chans,
        img_h=90,
        img_w=109,
        patch_size=10,
        embed_dim=128,
        num_heads=4,
        depth_spatial=2,
        depth_temporal=2,
        out_dim=64,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # patch 数，按 floor 计算
        self.num_patches = (img_h // patch_size) * (img_w // patch_size)

        # spatial CLS token
        self.spatial_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim)
        )

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer,
            num_layers=depth_spatial
        )

        # temporal CLS token
        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, 1 + 64, embed_dim)  # 先给大一点，forward里切片
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=depth_temporal
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # -------- spatial ViT --------
        x = x.reshape(B * T, C, H, W)
        x, Hp, Wp = self.patch_embed(x)  # (B*T, N, D)

        cls_tok = self.spatial_cls.expand(B * T, -1, -1)  # (B*T, 1, D)
        x = torch.cat([cls_tok, x], dim=1)                # (B*T, 1+N, D)

        pos = self.spatial_pos[:, :x.size(1), :]
        x = self.dropout(x + pos)

        x = self.spatial_encoder(x)                       # (B*T, 1+N, D)

        frame_feat = x[:, 0, :]                           # (B*T, D)
        frame_feat = frame_feat.view(B, T, -1)            # (B, T, D)

        # -------- temporal Transformer --------
        t_cls = self.temporal_cls.expand(B, -1, -1)       # (B, 1, D)
        x_t = torch.cat([t_cls, frame_feat], dim=1)       # (B, 1+T, D)

        t_pos = self.temporal_pos[:, :x_t.size(1), :]
        x_t = self.dropout(x_t + t_pos)

        x_t = self.temporal_encoder(x_t)                  # (B, 1+T, D)

        sat_feat = x_t[:, 0, :]                           # (B, D)

        return self.head(sat_feat)                        # (B, out_dim)


# =====================================================
# PV GRU encoder
# =====================================================

class TabularEncoder(nn.Module):

    def __init__(self, F):

        super().__init__()

        self.gru = nn.GRU(
            input_size=F,
            hidden_size=64,
            batch_first=True
        )

    def forward(self,x):

        # x (B,T,F)

        _,h = self.gru(x)

        return h.squeeze(0)

# =====================================================
# Forecast encoder
# =====================================================

class ForecastEncoder(nn.Module):

    def __init__(self,F):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(F,64),
            nn.ReLU(),

            nn.Linear(64,64)
        )

    def forward(self,x):

        # x (B,T,F)

        x = x.mean(dim=1)

        return self.mlp(x)

# =====================================================
# Full model
# =====================================================

class PVModel(nn.Module):

    def __init__(self,F_tab,F_fcst,C_sat):

        super().__init__()

        self.tab_encoder = TabularEncoder(F_tab)

        #self.sat_encoder = SatelliteEncoder(C_sat)
        self.sat_encoder = ViTSatelliteEncoder(
            in_chans=C_sat,
            img_h=90,
            img_w=109,
            patch_size=10,
            embed_dim=128,
            num_heads=4,
            depth_spatial=2,
            depth_temporal=2,
            out_dim=64,
            dropout=0.1
        )

        #self.fcst_encoder = ForecastEncoder(F_fcst) # test only

        self.fusion = nn.Sequential(
            #nn.Linear(64+64+64,128), # test only
            nn.Linear(64+64,128),
            nn.ReLU(),

            nn.Linear(128,1)
        )

    def forward(self,X_tab,X_sat,X_fcst):

        z_tab = self.tab_encoder(X_tab)

        z_sat = self.sat_encoder(X_sat)

        #z_fcst = self.fcst_encoder(X_fcst) # test only

        #z = torch.cat([z_tab,z_sat,z_fcst],dim=1) # test only
        z = torch.cat([z_tab,z_sat],dim=1)

        y = self.fusion(z)

        return y.squeeze(-1)

# =====================================================
# Train function
# =====================================================

def train_model(model,train_loader,test_loader,capacity,epochs=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)

    loss_fn = nn.MSELoss()

    writer = SummaryWriter("runs/pv_short")

    for epoch in range(epochs):

        model.train()

        train_losses = []

        for X_tab,X_sat,X_sat_mask,X_fcst,y in train_loader:
            # test only
            #print("tab mean:", X_tab.mean().item())
            #print("tab std:", X_tab.std().item())
            #print("sat mean and std:", X_sat[:, :, 0].mean(), X_sat[:, :, 0].std())
            X_tab = X_tab.to(device)
            X_sat = X_sat.to(device)
            X_fcst = X_fcst.to(device)
            y = y.to(device)

            pred = model(X_tab,X_sat,X_fcst)
            '''
            print("Detecting NaN in training")
            if torch.isnan(X_tab).any():
                print("NaN detected in X_tab")
            else:
                print("X_tab good")
            if torch.isnan(X_sat).any():
                print("NaN detected in X_sat")
            else:
                print("X_sat good")
            if torch.isnan(pred).any():
                print("NaN detected in pred")
            else:
                print("pred good")
            '''
            loss = loss_fn(pred,y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # =============================
        # evaluation
        # =============================

        model.eval()
        print("Init Eval")
        y_true = []
        y_pred = []

        with torch.no_grad():

            for X_tab,X_sat,X_sat_mask,X_fcst,y in test_loader:
                X_tab = X_tab.to(device)
                X_sat = X_sat.to(device)
                X_fcst = X_fcst.to(device)

                pred = model(X_tab,X_sat,X_fcst)

                '''
                print("Detecting NaN in eval")
                if torch.isnan(X_tab).any():
                    print("NaN detected in X_tab")
                else:
                    print("X_tab good")
                if torch.isnan(X_sat).any():
                    print("NaN detected in X_sat")
                else:
                    print("X_sat good")
                if torch.isnan(X_fcst).any():
                    print("NaN detected in X_fcst")
                if torch.isnan(pred).any():
                    print("NaN detected in pred")
                else:
                    print("pred good")
                '''
                    
                y_true.append(y.cpu().numpy())
                y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        print("y_true nan:", np.isnan(y_true).sum())
        print("y_pred nan:", np.isnan(y_pred).sum())

        mae = mean_absolute_error(y_true,y_pred)
        rmse = np.sqrt(mean_squared_error(y_true,y_pred))

        writer.add_scalar("train/loss",train_loss,epoch)
        writer.add_scalar("test/MAE",mae,epoch)
        writer.add_scalar("test/RMSE",rmse,epoch)

        print(f"Epoch {epoch}")
        print("Train loss:",train_loss)
        print("MAE:",mae,"acc:",1-mae/capacity)
        print("RMSE:",rmse,"acc:",1-rmse/capacity)
        print()

# =====================================================
# main
# =====================================================

if __name__ == "__main__":

    capacity = 0.5 * 117 * 1e3

    loader = LuoyangDataLoader(
        csv_path=Path("../datasets/luoyang_agg.csv"),
        feature_cols=[
            "total_active_kw",
            "mean_inner_temp",
            "status_ok"
        ],
        add_time_encoding=True,
        # solar_forecast_path="/home/weize/ai4energy/luoyang_demo/datasets/112.285_34.700_UTC0_model_solar_v5.csv",
        # wind_forecast_path="/home/weize/ai4energy/luoyang_demo/datasets/112.285_34.700_UTC0_model_wind_v5.csv",
        # forecast_feature_config={
        #     "solar": [
        #         "ssrd"
        #     ],
        #     "wind": [
        #         #"t2m",
        #         #"msl",
        #         "u10",
        #         "v10",
        #         "wind_speed_10m",
        #         #"u100",
        #         #"v100",
        #         #"wind_speed_100m",
        #     ]
        # },
        use_satellite=False, # save time
        satellite_root="/home/weize/ai4energy_crop",
        satellite_cache="/home/weize/sat_cache"
    )

    print("Building dataset...")

    X_tab,X_fcst,y,curr_gt,t = loader.make_short_dataset(
        history_len=16,
        horizon_hours=4
    )

    print("Dataset shapes")
    print("X_tab",X_tab.shape)
    print("X_fcst",X_fcst.shape)
    print("y",y.shape)
    # ======================================
    # sort by time
    # ======================================

    order = np.argsort(pd.to_datetime(t).values)

    X_tab = X_tab[order]
    X_fcst = X_fcst[order]
    y = y[order]

    # ======================================
    # split
    # ======================================

    split = int(len(y)*0.7)

    # compute normalization stats
    # tabular
    tab_mean = X_tab[:split].mean(axis=(0,1))
    tab_std  = X_tab[:split].std(axis=(0,1)) + 1e-6
    # forecast
    fcst_mean = X_fcst[:split].mean(axis=(0,1))
    fcst_std  = X_fcst[:split].std(axis=(0,1)) + 1e-6
    print("tab_mean", tab_mean)
    print("tab_std", tab_std)
    print("fcst_mean", fcst_mean)
    print("fcst_std", fcst_std)
    # satellite stats (sample)
    sat_mean = np.zeros(len(loader.satellite_vars)+1)
    sat_std  = np.zeros(len(loader.satellite_vars)+1)
    count = 0
    sample_n = min(2000, split)
    for i in range(sample_n):
        sat_seq, sat_mask = loader._build_satellite_sequence(pd.to_datetime(t[i]))
        valid = sat_mask.astype(bool)
        if valid.sum() == 0:
            continue
        sat_valid = sat_seq[valid]   # (Tv,C,H,W)
        sat_mean += sat_valid.mean(axis=(0,2,3))
        sat_std  += sat_valid.std(axis=(0,2,3))
        count += 1
    sat_mean /= count
    sat_std  /= count
    sat_std += 1e-6
    print("sat_mean", sat_mean)
    print("sat_std", sat_std)


    train_dataset = PVShortDataset(
        loader,
        X_tab[:split],
        X_fcst[:split],
        y[:split],
        t[:split],
        tab_mean,
        tab_std,
        fcst_mean,
        fcst_std,
        sat_mean,
        sat_std
    )

    test_dataset = PVShortDataset(
        loader,
        X_tab[split:],
        X_fcst[split:],
        y[split:],
        t[split:],
        tab_mean,
        tab_std,
        fcst_mean,
        fcst_std,
        sat_mean,
        sat_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4
    )

    model = PVModel(
        F_tab=X_tab.shape[2],
        F_fcst=X_fcst.shape[2],
        C_sat=len(loader.satellite_vars) + 1,
    )
    print("len(loader.satellite_vars) ", len(loader.satellite_vars))
    print(model)

    train_model(
        model,
        train_loader,
        test_loader,
        capacity,
        epochs=500
    )