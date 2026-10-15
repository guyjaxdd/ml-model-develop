import joblib
import streamlit as st
import pandas as pd

pipeline = joblib.load("./fps_model_Neural_Network.pkl")
st.set_page_config("FPS Prediction", layout="centered")

GAME_ENGINE_MAP = {
    "apexLegends": "Source",
    "arkSurvivalEvolved": "Unreal Engine 4",
    "battlefield1": "Frostbite 3",
    "battlefield4": "Frostbite 3",
    "callOfDutyBlackOps4": "IW engine",
    "counterStrikeGlobalOffensive": "Source",
    "dota2": "Source 2",
    "fortnite": "Unreal Engine 5",
    "grandTheftAuto5": "RAGE",
    "leagueOfLegends": "Hextech Engine",
    "minecraft": "lwjgl",
    "overwatch": "Prometheus",
    "playerUnknownsBattlegrounds": "Unreal Engine 4",
    "rocketLeague": "Unreal Engine 3",
    "rust": "Bevy",
    "worldOfTanks": "CoreX",
}

INTEGRATED_GPU_ARCHITECTURES = [
    "Generation 7.0", "Generation 7.5", "Generation 8.0",
    "Generation 9.0", "Generation 9.5", "GCN 5.0", "TeraScale",
    "Tesla", "Curie", "R400",
]
DEDICATED_GPU_ARCHITECTURES = [
    "Fermi", "Fermi 2.0", "GCN 1.0", "GCN 2.0", "GCN 3.0", "GCN 4.0",
    "GCN 5.1", "Kepler", "Kepler 2.0", "Maxwell", "Maxwell 2.0",
    "Pascal", "RDNA 1.0", "TeraScale 2", "TeraScale 3", "Tesla 2.0",
    "Turing", "Volta",
]


def predict(cpu_freq, cpu_turbo, gpu_arch, gpu_bw, gpu_boost, gpu_mem, gpu_pixel, gpu_rops,
            game_name, game_res, game_setting, game_engine):
    features = {
        "CpuFrequency": cpu_freq,
        "CpuTurboClock": cpu_turbo,
        "GpuArchitecture": gpu_arch,
        "GpuBandwidth": gpu_bw,
        "GpuBoostClock": gpu_boost,
        "GpuMemorySize": gpu_mem,
        "GpuPixelRate": gpu_pixel,
        "GpuNumberOfROPs": gpu_rops,
        "GameName": game_name,
        "GameResolution": game_res,
        "GameSetting": game_setting,
        "GameEngine": game_engine,
    }
    features = pd.DataFrame([features])
    result = pipeline.predict(features)[0]
    # A real computer can never run below 0 FPS. Linear models like Ridge
    # can still output negative numbers when extrapolating on extreme
    # inputs, so the result is floored here regardless of which model
    # is loaded.
    result = max(result, 0.0)
    return str(round(result, 1))


st.markdown("## FPS Prediction")
st.caption("Predict average FPS from GPU/CPU specs and game settings.")
st.markdown("""
**How to use this**
1. Pick your **GPU Architecture** and the **Game** you want to check.
2. Choose **Resolution** and **Graphics Setting** (Engine is set automatically based on the game).
3. Enter your CPU and GPU numbers (check your hardware's spec sheet or
   a tool like GPU-Z / CPU-Z if you're not sure of exact values).
4. Click **Predict** to see the estimated FPS.

No separate graphics card? Check **"No dedicated GPU"** below — bandwidth
and memory size are correctly 0 for integrated graphics, and the GPU
Architecture list will switch to integrated-only options.
""")

no_dedicated_gpu = st.checkbox(
    "No dedicated GPU (integrated graphics only)",
    help="Check this if your computer uses onboard/integrated graphics "
         "(e.g. Intel UHD/Iris, AMD APU) instead of a separate graphics card.",
)

col1, col2 = st.columns(2)
with col1:
    if no_dedicated_gpu:
        gpu_arch = st.selectbox("GPU Architecture (integrated)", INTEGRATED_GPU_ARCHITECTURES)
    else:
        gpu_arch = st.selectbox("GPU Architecture", DEDICATED_GPU_ARCHITECTURES)
with col2:
    game_name = st.selectbox("Game", ['apexLegends', 'arkSurvivalEvolved', 'battlefield1', 'battlefield4',
        'callOfDutyBlackOps4', 'counterStrikeGlobalOffensive', 'dota2',
        'fortnite', 'grandTheftAuto5', 'leagueOfLegends', 'minecraft',
        'overwatch', 'playerUnknownsBattlegrounds', 'rocketLeague', 'rust',
        'worldOfTanks'])

col3, col4 = st.columns(2)
with col3:
    game_res = st.selectbox("Resolution", [720, 1080, 1440])
with col4:
    game_setting = st.selectbox("Graphics Setting", ['low', 'med', 'high', 'max'])

game_engine = GAME_ENGINE_MAP[game_name]
st.caption(f"Engine: **{game_engine}** (set automatically from the game)")

col5, col6 = st.columns(2)
with col5:

    cpu_freq = st.number_input("CPU Frequency (MHz)", min_value=1500, max_value=5000, value=3500, step=50)
with col6:
    cpu_turbo = st.number_input("CPU Turbo Clock (MHz)", min_value=1500, max_value=5500, value=4000, step=50)

col7, col8 = st.columns(2)
with col7:
    if no_dedicated_gpu:

        gpu_bw = st.number_input("GPU Bandwidth (MB/s)", value=0, disabled=True)
    else:
        gpu_bw = st.number_input("GPU Bandwidth (MB/s)", min_value=1000, max_value=1100000, value=224400, step=100)
with col8:

    gpu_boost = st.number_input("GPU Boost Clock (MHz)", min_value=300, max_value=2200, value=1582, step=10)

col9, col10 = st.columns(2)
with col9:
    if no_dedicated_gpu:
        gpu_mem = st.number_input("GPU Memory Size (MB)", value=0, disabled=True)
    else:
        gpu_mem = st.number_input("GPU Memory Size (MB)", min_value=256, max_value=25000, value=6000, step=500)
with col10:

    gpu_pixel = st.number_input("GPU Pixel Rate", min_value=400, max_value=175000, value=82030, step=100)


gpu_rops = st.number_input("GPU Number of ROPs", min_value=1, max_value=100, value=48, step=1)

if st.button("Predict"):
    fps = predict(cpu_freq, cpu_turbo, gpu_arch, gpu_bw, gpu_boost, gpu_mem, gpu_pixel, gpu_rops,
                  game_name, game_res, game_setting, game_engine)
    st.text_input("Predicted FPS", value=fps, disabled=True)
