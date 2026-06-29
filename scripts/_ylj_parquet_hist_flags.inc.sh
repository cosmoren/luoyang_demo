# Shared Parquet history feature flags for YLJ train/test scripts.
# Sets PARQUET_HIST_FLAGS (space-separated CLI args). Mutates TAG when called.
#
# Env defaults (override before sourcing):
#   USE_ALL=0  -> --ylj_parquet_use_all (all 14 hist channels)
#   USE_GHI=1, USE_GHI_SOLARGIS=1, USE_TEMP_SOLARGIS=1
#   USE_KT_RAMP=0, USE_GHI_RAMP=0, USE_GHI_ROLL_MEAN=0, USE_GHI_ROLL_STD=0
#   USE_OM_CLOUD_PCT=0, USE_OM_CLOUD_PCT_LOW_MID=0
#   USE_WS_SOLARGIS=0, USE_WD_SOLARGIS=0, USE_PREC_SOLARGIS=0
#   USE_PWAT_SOLARGIS=0, USE_SDWE_SOLARGIS=0

ylj_build_parquet_hist_flags() {
  USE_ALL=${USE_ALL:-0}
  USE_GHI=${USE_GHI:-0}
  USE_GHI_SOLARGIS=${USE_GHI_SOLARGIS:-0}
  USE_TEMP_SOLARGIS=${USE_TEMP_SOLARGIS:-0}
  USE_KT_RAMP=${USE_KT_RAMP:-0}
  USE_GHI_RAMP=${USE_GHI_RAMP:-0}
  USE_GHI_ROLL_MEAN=${USE_GHI_ROLL_MEAN:-0}
  USE_GHI_ROLL_STD=${USE_GHI_ROLL_STD:-0}
  USE_OM_CLOUD_PCT=${USE_OM_CLOUD_PCT:-0}
  USE_OM_CLOUD_PCT_LOW_MID=${USE_OM_CLOUD_PCT_LOW_MID:-0}
  USE_WS_SOLARGIS=${USE_WS_SOLARGIS:-0}
  USE_WD_SOLARGIS=${USE_WD_SOLARGIS:-0}
  USE_PREC_SOLARGIS=${USE_PREC_SOLARGIS:-0}
  USE_PWAT_SOLARGIS=${USE_PWAT_SOLARGIS:-0}
  USE_SDWE_SOLARGIS=${USE_SDWE_SOLARGIS:-0}

  PARQUET_HIST_FLAGS=""
  if [ "$USE_ALL" = "1" ]; then
    PARQUET_HIST_FLAGS="--ylj_parquet_use_all"
    TAG="${TAG}_all"
    return
  fi

  [ "$USE_GHI" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_ghi"; TAG="${TAG}_ghi"; }
  [ "$USE_GHI_SOLARGIS" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_ghi_solargis"; TAG="${TAG}_ghi_sg"; }
  [ "$USE_TEMP_SOLARGIS" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_temp_solargis"; TAG="${TAG}_temp_sg"; }
  [ "$USE_KT_RAMP" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_kt_ramp"; TAG="${TAG}_kt_ramp"; }
  [ "$USE_GHI_RAMP" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_ghi_ramp"; TAG="${TAG}_ghi_ramp"; }
  [ "$USE_GHI_ROLL_MEAN" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_ghi_roll_mean"; TAG="${TAG}_roll_mean"; }
  [ "$USE_GHI_ROLL_STD" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_ghi_roll_std"; TAG="${TAG}_roll_std"; }
  [ "$USE_OM_CLOUD_PCT" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_om_cloud_pct"; TAG="${TAG}_cloud"; }
  [ "$USE_OM_CLOUD_PCT_LOW_MID" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_om_cloud_pct_low_mid"; TAG="${TAG}_cloud_lm"; }
  [ "$USE_WS_SOLARGIS" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_ws_solargis"; TAG="${TAG}_ws_sg"; }
  [ "$USE_WD_SOLARGIS" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_wd_solargis"; TAG="${TAG}_wd_sg"; }
  [ "$USE_PREC_SOLARGIS" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_prec_solargis"; TAG="${TAG}_prec_sg"; }
  [ "$USE_PWAT_SOLARGIS" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_pwat_solargis"; TAG="${TAG}_pwat_sg"; }
  [ "$USE_SDWE_SOLARGIS" = "1" ] && { PARQUET_HIST_FLAGS+=" --ylj_parquet_sdwe_solargis"; TAG="${TAG}_sdwe_sg"; }
  PARQUET_HIST_FLAGS="${PARQUET_HIST_FLAGS# }"
}
