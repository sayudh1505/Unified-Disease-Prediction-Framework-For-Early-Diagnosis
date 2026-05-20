import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# -----------------------------
# Page configuration (must be first)
# -----------------------------
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# -----------------------------
# Resolve paths (edit these if needed)
# -----------------------------
diabetes_path = "diabetes_model.sav"
heart_path    = "heart_disease_model.sav"
pd_path       = "parkinsons_model.sav"

def load_pickle_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------
# Load models
# -----------------------------
diabetes_model = load_pickle_model("diabetes_model.sav")
heart_disease_model = load_pickle_model("heart_disease_model.sav")
parkinsons_model = load_pickle_model("parkinsons_model.sav")

# -----------------------------
# Sidebar navigation
# -----------------------------
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Check", "Heart Risk Check", "Parkinson's Voice Check"],
        menu_icon="hospital-fill",
        icons=["activity", "heart", "person"],
        default_index=0
    )

# Small helper to show results nicely
def show_result(msg: str):
    if "not" in msg.lower():
        st.success(msg)
    else:
        st.error(msg)

# =========================================================
# Diabetes (Plain-English)
# =========================================================
if selected == "Diabetes Check":
    st.title("Diabetes Check (Type-2 risk)")

    # Add mode toggle
    mode_col1, mode_col2 = st.columns([3, 1])
    with mode_col2:
        diabetes_mode = st.selectbox(
            "Mode",
            ["Normal", "Advanced"],
            key="diabetes_mode",
            help="Normal: Essential fields only | Advanced: All parameters"
        )

    if diabetes_mode == "Normal":
        st.info("Fill in essential details below. These are the key indicators for diabetes risk.")
        col1, col2, col3 = st.columns(3)

        with col1:
            Glucose = st.number_input(
                "Blood sugar (mg/dL)",
                min_value=0.0, step=1.0,
                help="Fasting blood glucose. Example: 90–120 mg/dL.",
                key="diabetes_glucose_normal"
            )
        with col2:
            BMI = st.number_input(
                "BMI (kg/m²)",
                min_value=0.0, step=0.1, format="%.1f",
                help="Body Mass Index. Example: 22.5.",
                key="diabetes_bmi_normal"
            )
        with col3:
            Age = st.number_input(
                "Age (years)",
                min_value=0, step=1, help="Age of the person.",
                key="diabetes_age_normal"
            )

        # Default values for advanced fields in normal mode
        Pregnancies = 0
        BloodPressure = 0
        SkinThickness = 0
        Insulin = 0
        DiabetesPedigreeFunction = 0.0

    else:  # Advanced mode
        st.info("Fill in all parameters below for detailed diabetes risk assessment.")
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input(
                "Pregnancies",
                min_value=0, step=1, help="How many times the person has been pregnant. If male, keep 0.",
                key="diabetes_pregnancies"
            )
        with col2:
            Glucose = st.number_input(
                "Blood sugar (mg/dL)",
                min_value=0.0, step=1.0,
                help="Fasting blood glucose. Example: 90–120 mg/dL.",
                key="diabetes_glucose_adv"
            )
        with col3:
            BloodPressure = st.number_input(
                "Blood pressure (mmHg)",
                min_value=0.0, step=1.0,
                help="Resting diastolic pressure. Example: 70–90 mmHg.",
                key="diabetes_bp_adv"
            )
        with col1:
            SkinThickness = st.number_input(
                "Skin fold thickness (mm)",
                min_value=0.0, step=0.5,
                help="Triceps skinfold (if known). Otherwise leave default.",
                key="diabetes_skin_adv"
            )
        with col2:
            Insulin = st.number_input(
                "Insulin (µU/mL)",
                min_value=0.0, step=1.0,
                help="2-hour serum insulin. Leave 0 if unknown.",
                key="diabetes_insulin_adv"
            )
        with col3:
            BMI = st.number_input(
                "BMI (kg/m²)",
                min_value=0.0, step=0.1, format="%.1f",
                help="Body Mass Index. Example: 22.5.",
                key="diabetes_bmi_adv"
            )
        with col1:
            DiabetesPedigreeFunction = st.number_input(
                "Family history score",
                min_value=0.0, step=0.01, format="%.2f",
                help="Higher = stronger family history of diabetes. If unknown, keep default.",
                key="diabetes_dpf_adv"
            )
        with col2:
            Age = st.number_input(
                "Age (years)",
                min_value=0, step=1, help="Age of the person.",
                key="diabetes_age_adv"
            )

    if st.button("Check diabetes risk", key="diabetes_button"):
        user_input = [
            Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
            BMI, DiabetesPedigreeFunction, Age
        ]
        try:
            pred = diabetes_model.predict([user_input])[0]
            msg = "The person is diabetic" if pred == 1 else "The person is not diabetic"
            show_result(msg)
        except Exception as e:
            st.error(f"Could not generate a prediction. Details: {e}")

# =========================================================
# Heart Disease (Plain-English with dropdowns)
# =========================================================
if selected == "Heart Risk Check":
    st.title("Heart Disease Risk Check")

    # Add mode toggle
    mode_col1, mode_col2 = st.columns([3, 1])
    with mode_col2:
        heart_mode = st.selectbox(
            "Mode",
            ["Normal", "Advanced"],
            key="heart_mode",
            help="Normal: Essential fields only | Advanced: All parameters"
        )

    # Mappings from friendly text -> model codes (kept consistent with your earlier 0–3 scheme)
    sex_map = {"Female": 0, "Male": 1}
    cp_map = {
        "Typical chest pain": 0,
        "Atypical chest pain": 1,
        "Non-chest pain discomfort": 2,
        "No symptoms": 3,
    }
    yesno_map = {"No": 0, "Yes": 1}
    restecg_map = {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Possible left ventricular hypertrophy": 2,
    }
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {
        "Normal": 0,
        "Fixed defect": 1,
        "Reversible defect": 2,
    }

    if heart_mode == "Normal":
        st.info("Answer in simple terms. We'll use these key indicators to assess your heart disease risk.")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (years)", min_value=0, step=1, help="Age of the person.", key="heart_age_normal")
        with col2:
            sex = st.selectbox("Biological sex", list(sex_map.keys()), key="heart_sex_normal")
        with col3:
            chol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, step=1.0, help="Total serum cholesterol.", key="heart_chol_normal")
        with col1:
            trestbps = st.number_input("Resting BP (mmHg)", min_value=0.0, step=1.0, help="Typical range 90–140.", key="heart_bp_normal")
        with col2:
            thalach = st.number_input("Max heart rate (bpm)", min_value=0.0, step=1.0, key="heart_rate_normal")
        with col3:
            cp = st.selectbox("Chest symptoms", list(cp_map.keys()), help="Describe the most fitting chest symptom.", key="heart_cp_normal")

        # Default values for advanced fields in normal mode
        fbs = "No"
        restecg = "Normal"
        exang = "No"
        oldpeak = 0.0
        slope = "Upsloping"
        ca = 0
        thal = "Normal"

    else:  # Advanced mode
        st.info("Answer in everyday terms. We'll convert to the numbers your model expects.")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (years)", min_value=0, step=1, help="Age of the person.", key="heart_age_adv")
        with col2:
            sex = st.selectbox("Biological sex", list(sex_map.keys()), key="heart_sex_adv")
        with col3:
            cp = st.selectbox("Chest symptoms", list(cp_map.keys()), help="Describe the most fitting chest symptom.", key="heart_cp_adv")
        with col1:
            trestbps = st.number_input("Resting BP (mmHg)", min_value=0.0, step=1.0, help="Typical range 90–140.", key="heart_bp_adv")
        with col2:
            chol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, step=1.0, help="Total serum cholesterol.", key="heart_chol_adv")
        with col3:
            fbs = st.selectbox("Fasting sugar > 120 mg/dL?", list(yesno_map.keys()), key="heart_fbs_adv")
        with col1:
            restecg = st.selectbox("ECG (resting)", list(restecg_map.keys()), key="heart_ecg_adv")
        with col2:
            thalach = st.number_input("Max heart rate (bpm)", min_value=0.0, step=1.0, key="heart_rate_adv")
        with col3:
            exang = st.selectbox("Chest pain with exercise?", list(yesno_map.keys()), key="heart_exang_adv")
        with col1:
            oldpeak = st.number_input("ST depression (exercise)", min_value=0.0, step=0.1, format="%.1f",
                                      help="Higher can indicate ischemia. If unknown, leave default.", key="heart_oldpeak_adv")
        with col2:
            slope = st.selectbox("ST segment slope", list(slope_map.keys()), key="heart_slope_adv")
        with col3:
            ca = st.number_input("# major vessels seen (0–3)", min_value=0, max_value=3, step=1,
                                 help="Number of vessels colored by fluoroscopy.", key="heart_ca_adv")
        with col1:
            thal = st.selectbox("Thallium stress test", list(thal_map.keys()), key="heart_thal_adv")

    if st.button("Check heart risk", key="heart_button"):
        user_input = [
            age,
            sex_map[sex],
            cp_map[cp],
            trestbps,
            chol,
            yesno_map[fbs],
            restecg_map[restecg],
            thalach,
            yesno_map[exang],
            oldpeak,
            slope_map[slope],
            ca,
            thal_map[thal],
        ]
        try:
            pred = heart_disease_model.predict([user_input])[0]
            msg = "The person is having heart disease" if pred == 1 else "The person does not have any heart disease"
            show_result(msg)
        except Exception as e:
            st.error(f"Could not generate a prediction. Details: {e}")

# =========================================================
# Parkinson's (Plain-English + explanations)
# =========================================================
if selected == "Parkinson's Voice Check":
    st.title("Parkinson's Voice Check")

    # Add mode toggle
    mode_col1, mode_col2 = st.columns([3, 1])
    with mode_col2:
        parkinsons_mode = st.selectbox(
            "Mode",
            ["Normal", "Advanced"],
            key="parkinsons_mode",
            help="Normal: Essential parameters only | Advanced: All 22 parameters"
        )

    if parkinsons_mode == "Normal":
        st.info(
            "This check uses key voice measurements from a short voice recording. "
            "Provide the main parameters that characterize voice quality."
        )
        col1, col2, col3 = st.columns(3)

        with col1:
            fo = st.number_input("Pitch (Fo, Hz)", min_value=0.0, step=0.1,
                                 help="Average voice pitch.", key="pk_fo_normal")
        with col2:
            Jitter_percent = st.number_input("Jitter (%)", min_value=0.0, step=0.0001, format="%.4f",
                                             help="Pitch instability (smaller = steadier).", key="pk_jitter_normal")
        with col3:
            Shimmer = st.number_input("Shimmer", min_value=0.0, step=0.0001, format="%.4f",
                                      help="Loudness variation.", key="pk_shimmer_normal")
        with col1:
            HNR = st.number_input("Harmonics–to–Noise (HNR)", min_value=0.0, step=0.01, format="%.2f",
                                  help="Higher = cleaner voice signal.", key="pk_hnr_normal")
        with col2:
            RPDE = st.number_input("RPDE", min_value=0.0, step=0.0001, format="%.4f",
                                   help="Signal complexity measure.", key="pk_rpde_normal")
        with col3:
            PPE = st.number_input("PPE", min_value=0.0, step=0.0001, format="%.4f",
                                  help="Pitch period entropy (stability).", key="pk_ppe_normal")

        # Default values for advanced fields in normal mode
        fhi = 0.0
        flo = 0.0
        Jitter_Abs = 0.0
        RAP = 0.0
        PPQ = 0.0
        DDP = 0.0
        Shimmer_dB = 0.0
        APQ3 = 0.0
        APQ5 = 0.0
        APQ = 0.0
        DDA = 0.0
        NHR = 0.0
        DFA = 0.0
        spread1 = 0.0
        spread2 = 0.0
        D2 = 0.0

    else:  # Advanced mode
        st.info(
            "This check uses measurements from a short voice recording (sustained vowel). "
            "If you don't have these exact values, leave defaults or use values calculated by your voice-analysis tool."
        )
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.number_input("Pitch (Fo, Hz)", min_value=0.0, step=0.1,
                                 help="Average voice pitch.", key="pk_fo_adv")
        with col2:
            fhi = st.number_input("Highest pitch (Hz)", min_value=0.0, step=0.1,
                                  help="Highest pitch detected.", key="pk_fhi_adv")
        with col3:
            flo = st.number_input("Lowest pitch (Hz)", min_value=0.0, step=0.1,
                                  help="Lowest pitch detected.", key="pk_flo_adv")
        with col4:
            Jitter_percent = st.number_input("Jitter (%)", min_value=0.0, step=0.0001, format="%.4f",
                                             help="Pitch instability percentage (smaller is steadier).", key="pk_jitter_adv")
        with col5:
            Jitter_Abs = st.number_input("Jitter (Abs)", min_value=0.0, step=0.00001, format="%.5f",
                                         help="Absolute pitch variation.", key="pk_jitter_abs_adv")
        with col1:
            RAP = st.number_input("RAP", min_value=0.0, step=0.0001, format="%.4f",
                                  help="Relative average perturbation (pitch).", key="pk_rap_adv")
        with col2:
            PPQ = st.number_input("PPQ", min_value=0.0, step=0.0001, format="%.4f",
                                  help="Pitch period variability.", key="pk_ppq_adv")
        with col3:
            DDP = st.number_input("DDP", min_value=0.0, step=0.0001, format="%.4f",
                                  help="Derived from jitter (steadiness).", key="pk_ddp_adv")
        with col4:
            Shimmer = st.number_input("Shimmer", min_value=0.0, step=0.0001, format="%.4f",
                                      help="Loudness variation.", key="pk_shimmer_adv")
        with col5:
            Shimmer_dB = st.number_input("Shimmer (dB)", min_value=0.0, step=0.01, format="%.2f",
                                         help="Loudness variation in dB.", key="pk_shimmer_db_adv")
        with col1:
            APQ3 = st.number_input("APQ3", min_value=0.0, step=0.0001, format="%.4f",
                                   help="Amplitude perturbation quotient (3 cycles).", key="pk_apq3_adv")
        with col2:
            APQ5 = st.number_input("APQ5", min_value=0.0, step=0.0001, format="%.4f",
                                   help="Amplitude perturbation quotient (5 cycles).", key="pk_apq5_adv")
        with col3:
            APQ = st.number_input("APQ", min_value=0.0, step=0.0001, format="%.4f",
                                  help="Amplitude perturbation quotient (overall).", key="pk_apq_adv")
        with col4:
            DDA = st.number_input("DDA", min_value=0.0, step=0.0001, format="%.4f",
                                  help="Related to shimmer (amplitude steadiness).", key="pk_dda_adv")
        with col5:
            NHR = st.number_input("Noise–to–Harmonics (NHR)", min_value=0.0, step=0.0001, format="%.4f",
                                  help="More noise can indicate rough voice.", key="pk_nhr_adv")
        with col1:
            HNR = st.number_input("Harmonics–to–Noise (HNR)", min_value=0.0, step=0.01, format="%.2f",
                                  help="Higher HNR = cleaner voice signal.", key="pk_hnr_adv")
        with col2:
            RPDE = st.number_input("RPDE", min_value=0.0, step=0.0001, format="%.4f",
                                   help="Signal complexity measure.", key="pk_rpde_adv")
        with col3:
            DFA = st.number_input("DFA", min_value=0.0, step=0.0001, format="%.4f",
                                  help="Fractal scaling of the signal.", key="pk_dfa_adv")
        with col4:
            spread1 = st.number_input("Spread1", step=0.0001, format="%.4f",
                                      help="Frequency variation summary (1).", key="pk_spread1_adv")
        with col5:
            spread2 = st.number_input("Spread2", step=0.0001, format="%.4f",
                                      help="Frequency variation summary (2).", key="pk_spread2_adv")
        with col1:
            D2 = st.number_input("D2", min_value=0.0, step=0.0001, format="%.4f",
                                 help="Signal dynamical complexity.", key="pk_d2_adv")
        with col2:
            PPE = st.number_input("PPE", min_value=0.0, step=0.0001, format="%.4f",
                                  help="Pitch period entropy (stability).", key="pk_ppe_adv")

    if st.button("Check parkisnons risk", key="parkinsons_button"):
        user_input = [
            fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
            Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE,
            DFA, spread1, spread2, D2, PPE
        ]
        try:
            pred = parkinsons_model.predict([user_input])[0]
            msg = "The person has Parkinson's disease" if pred == 1 else "The person does not have Parkinson's disease"
            show_result(msg)
        except Exception as e:
            st.error(f"Could not generate a prediction. Details: {e}")
