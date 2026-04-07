# ============================================================
#  AgriML India — Smart Farming Intelligence
#  streamlit run agriml_app.py
#  pip install streamlit plotly pandas scikit-learn openpyxl
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AgriML India",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── sklearn ──────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Montserrat:wght@700;800;900&display=swap');
html,body,[class*="css"]{ font-family:'Nunito',sans-serif !important; }
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMainBlockContainer"],
.block-container,.main,.main>div,section.main {
    background:#f8faf8 !important;
}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#1a3d12 0%,#2d6a1f 100%) !important;
}
[data-testid="stSidebar"] *{ color:#fff !important; }
[data-testid="stSidebar"] .stRadio label{ color:#e0f0e0 !important; font-size:13px; }
[data-testid="stSidebar"] .stSelectbox label{ color:#a5d6a7 !important; }
.stMetric{ background:#fff; border:1px solid #d6e8d6; border-radius:12px; padding:12px; }
.stButton>button{ background:linear-gradient(135deg,#2d6a1f,#3d8a28);
    color:#fff; border:none; border-radius:10px; font-weight:700; padding:10px 20px; }
.stButton>button:hover{ background:linear-gradient(135deg,#3d8a28,#4caf50); }
.stTabs [data-baseweb="tab"]{
    background:#e8f5e9; border-radius:8px 8px 0 0;
    color:#1a3d12; font-weight:700;
}
.stTabs [aria-selected="true"]{
    background:#2d6a1f !important; color:#fff !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════
BL = ["Chinnasalem","Cuddalore","Kallakurichi","Kurinjipadi","Panruti","Sankarapuram"]
VA = ["Co_43","Delux Ponni","Ponmani"]
ST = ["alluvial","clay"]
NU = ["dry","wet"]
YH = {1:5757,2:11487,3:17266,4:24473,5:31046,6:36228}
BV = {"Cuddalore":"Co_43","Kallakurichi":"Co_43","Sankarapuram":"Co_43",
      "Chinnasalem":"Delux Ponni","Panruti":"Delux Ponni","Kurinjipadi":"Ponmani"}
G  = ["#2d6a1f","#3a8a28","#4caf50","#66bb6a","#81c784","#a5d6a7"]

# ═══════════════════════════════════════════════════════════════
#  MODEL TRAINING — cached, runs once on startup
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🌾 Training ML models on your data…")
def train_all_models(csv_path="paddydataset.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None

    df.columns = df.columns.str.strip()
    df["Variety"]    = df["Variety"].str.strip().str.title()
    df["Agriblock"]  = df["Agriblock"].str.strip()
    df["Soil Types"] = df["Soil Types"].str.strip()
    df["Nursery"]    = df["Nursery"].str.strip()
    df = df.drop_duplicates().reset_index(drop=True)

    df_enc  = df.copy()
    le_dict = {}
    for col in df_enc.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        le_dict[col] = le

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # ── P1: Yield Prediction ─────────────────────────────────
    X1 = df_enc.drop("Paddy yield(in Kg)", axis=1)
    y1 = df_enc["Paddy yield(in Kg)"]
    Xtr1,Xte1,ytr1,yte1 = train_test_split(X1,y1,test_size=0.2,random_state=42)
    m1 = RandomForestRegressor(n_estimators=200,max_depth=15,
                                min_samples_leaf=2,random_state=42)
    m1.fit(Xtr1,ytr1)
    p1 = m1.predict(Xte1)
    cv1 = cross_val_score(m1,X1,y1,cv=kf,scoring="neg_mean_absolute_error")
    results["p1"] = {
        "model":m1,"features":list(X1.columns),
        "mae":mean_absolute_error(yte1,p1),
        "r2":r2_score(yte1,p1),
        "cv_mae":-cv1.mean(),"cv_std":cv1.std()
    }

    # ── P2: Variety Recommendation ───────────────────────────
    var_le = LabelEncoder()
    y2     = var_le.fit_transform(df["Variety"])
    X2 = df_enc.drop(columns=["Variety","Paddy yield(in Kg)"],errors="ignore")
    Xtr2,Xte2,ytr2,yte2 = train_test_split(X2,y2,test_size=0.2,
                                             stratify=y2,random_state=42)
    m2 = RandomForestClassifier(n_estimators=200,max_depth=15,
                                 min_samples_leaf=2,random_state=42)
    m2.fit(Xtr2,ytr2)
    cv2 = cross_val_score(m2,X2,y2,cv=kf,scoring="accuracy")
    results["p2"] = {
        "model":m2,"le":var_le,"features":list(X2.columns),
        "acc":accuracy_score(yte2,m2.predict(Xte2)),
        "cv_acc":cv2.mean(),"cv_std":cv2.std(),
        "labels":list(var_le.classes_)
    }

    # ── P3: Fertilizer Optimization ──────────────────────────
    fert_targets = ["DAP_20days","Weed28D_thiobencarb","Urea_40Days",
                    "Potassh_50Days","Micronutrients_70Days","Pest_60Day(in ml)"]
    fert_targets = [c for c in fert_targets if c in df_enc.columns]
    non_feat = fert_targets + ["Paddy yield(in Kg)"]
    feat3 = [c for c in df_enc.columns if c not in non_feat]
    X3 = df_enc[feat3]
    fert_models, fert_accs = {}, []
    for t in fert_targets:
        y3 = df_enc[t].round(2).astype(str)
        Xtr3,Xte3,ytr3,yte3 = train_test_split(X3,y3,test_size=0.2,
                                                 stratify=y3,random_state=42)
        clf = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
        clf.fit(Xtr3,ytr3)
        acc = accuracy_score(yte3,clf.predict(Xte3))
        fert_models[t] = clf
        fert_accs.append(acc)
    results["p3"] = {
        "models":fert_models,"features":feat3,"targets":fert_targets,
        "mean_acc":float(np.mean(fert_accs))
    }

    # ── P4: Pre-Harvest ──────────────────────────────────────
    X4 = df[["Trash(in bundles)","Hectares"]]
    y4 = df["Paddy yield(in Kg)"]
    Xtr4,Xte4,ytr4,yte4 = train_test_split(X4,y4,test_size=0.2,random_state=42)
    m4 = RandomForestRegressor(n_estimators=200,max_depth=15,
                                min_samples_leaf=2,random_state=42)
    m4.fit(Xtr4,ytr4)
    p4 = m4.predict(Xte4)
    cv4 = cross_val_score(m4,X4,y4,cv=kf,scoring="neg_mean_absolute_error")
    results["p4"] = {
        "model":m4,
        "mae":mean_absolute_error(yte4,p4),
        "r2":r2_score(yte4,p4),
        "cv_mae":-cv4.mean(),"cv_std":cv4.std()
    }

    # ── P5: Anomaly Detection ─────────────────────────────────
    scaler = StandardScaler()
    X5 = scaler.fit_transform(df_enc)
    m5 = IsolationForest(contamination=0.046,n_estimators=200,random_state=42)
    labels5 = m5.fit_predict(X5)
    anomaly_count = int((labels5 == -1).sum())
    pca5 = PCA(n_components=2, random_state=42)
    X5_2d = pca5.fit_transform(X5)
    df5_vis = pd.DataFrame({
        "PC1":X5_2d[:,0],"PC2":X5_2d[:,1],
        "Label":["Anomaly" if l==-1 else "Normal" for l in labels5],
        "Yield":df["Paddy yield(in Kg)"].values,
        "Agriblock":df["Agriblock"].values
    })
    results["p5"] = {
        "model":m5,"scaler":scaler,"anomaly_count":anomaly_count,
        "total":len(df),"df_vis":df5_vis
    }

    results["df"]     = df
    results["df_enc"] = df_enc
    results["le_dict"]= le_dict
    results["trained"]= True
    return results

# ═══════════════════════════════════════════════════════════════
#  LANGUAGE TRANSLATIONS
# ═══════════════════════════════════════════════════════════════
LANGS = ["English","தமிழ்","हिंदी","తెలుగు","മലയാളം"]

T = {
  "English":{
    "title":"AgriML India — Smart Farming",
    "pages":["🏠 Dashboard","📊 Data Explorer","⚖️ Yield Predictor",
             "🌿 Variety Advisor","🧪 Fertilizer Planner","🔮 Pre-Harvest",
             "🚨 Risk Monitor","💬 AgriBot","🔢 Manual Predict"],
    "block":"Agriblock","variety":"Variety","hectares":"Farm Size (Ha)",
    "soil":"Soil Type","nursery":"Nursery","bundles":"Trash Bundles",
    "dap":"DAP Day 20 (Kg)","urea":"Urea Day 40 (Kg)",
    "potash":"Potash Day 50 (Kg)","micro":"Micronutrients Day 70 (Kg)",
    "weed":"Weed Control Day 28 (Kg)","pest":"Pest Control Day 35 (ml)",
    "actual_yield":"Your Actual Yield (Kg)",
    "predict":"🔮 Predict","recommend":"🌿 Recommend",
    "calc_dose":"🧪 Calculate Doses","estimate":"📦 Estimate Yield",
    "check_risk":"🚨 Check Risk",
    "predicted_yield":"Predicted Yield","recommended_variety":"Recommended Variety",
    "fertilizer_doses":"Fertilizer Doses","estimated_yield":"Estimated Yield",
    "risk_status":"Risk Status",
    "excellent":"✅ EXCELLENT — Above average! Keep it up!",
    "average":"🟡 AVERAGE — Can improve 15-20%",
    "underperform":"⚠️ UNDERPERFORMING — Below expected. Fix inputs!",
    "safe":"✅ SAFE — Farm performing well",
    "risky":"⚠️ RISK DETECTED — Action needed",
    "p1_desc":"Predicts exact paddy yield from all 45 farm features",
    "p2_desc":"Recommends best paddy variety for your block and soil",
    "p3_desc":"Calculates optimal doses for 6 fertilizers",
    "p4_desc":"Estimates yield by counting trash bundles before harvest",
    "p5_desc":"Detects underperforming farms using anomaly detection",
    "ask_me":"Ask me anything...",
    "clear_chat":"🔄 Clear Chat",
  },
  "தமிழ்":{
    "title":"AgriML India — திறமையான விவசாயம்",
    "pages":["🏠 டாஷ்போர்டு","📊 தரவு ஆய்வு","⚖️ மகசூல் கணிப்பு",
             "🌿 ரக ஆலோசனை","🧪 உர திட்டமிடல்","🔮 முன் அறுவடை",
             "🚨 ஆபத்து கண்காணிப்பு","💬 AgriBot","🔢 கைமுறை கணிப்பு"],
    "block":"விவசாய பகுதி","variety":"ரகம்","hectares":"பண்ணை அளவு (ஹெ)",
    "soil":"மண் வகை","nursery":"நாற்றங்கால்","bundles":"கட்டுகள் எண்ணிக்கை",
    "dap":"DAP நாள் 20 (கி)","urea":"யூரியா நாள் 40 (கி)",
    "potash":"பொட்டாஷ் நாள் 50 (கி)","micro":"நுண்ணூட்டம் நாள் 70 (கி)",
    "weed":"களை கட்டுப்பாடு நாள் 28 (கி)","pest":"பூச்சி நாள் 35 (மி.லி)",
    "actual_yield":"உண்மை மகசூல் (கி)",
    "predict":"🔮 கணிக்கவும்","recommend":"🌿 பரிந்துரை",
    "calc_dose":"🧪 அளவு கணக்கிடு","estimate":"📦 மதிப்பீடு",
    "check_risk":"🚨 ஆபத்து சரிபார்",
    "predicted_yield":"கணிக்கப்பட்ட மகசூல்","recommended_variety":"பரிந்துரைக்கப்பட்ட ரகம்",
    "fertilizer_doses":"உர அளவுகள்","estimated_yield":"மதிப்பீட்டு மகசூல்",
    "risk_status":"ஆபத்து நிலை",
    "excellent":"✅ சிறப்பானது — சராசரிக்கு மேல்!",
    "average":"🟡 சராசரி — 15-20% மேம்படலாம்",
    "underperform":"⚠️ குறைவு — எதிர்பார்ப்புக்கு கீழே!",
    "safe":"✅ பாதுகாப்பு — நல்ல செயல்திறன்",
    "risky":"⚠️ ஆபத்து — நடவடிக்கை எடுங்கள்",
    "p1_desc":"45 பண்ணை அம்சங்களிலிருந்து மகசூலை கணிக்கிறது",
    "p2_desc":"உங்கள் பகுதி மற்றும் மண்ணிற்கு சிறந்த ரகம் பரிந்துரைக்கிறது",
    "p3_desc":"6 உரங்களுக்கு உகந்த அளவுகளை கணக்கிடுகிறது",
    "p4_desc":"அறுவடைக்கு முன் கட்டு எண்ணி மகசூல் மதிப்பீடு",
    "p5_desc":"குறைவான செயல்திறன் கொண்ட பண்ணைகளை கண்டறிகிறது",
    "ask_me":"கேளுங்கள்...","clear_chat":"🔄 அழி",
  },
  "हिंदी":{
    "title":"AgriML India — स्मार्ट खेती",
    "pages":["🏠 डैशबोर्ड","📊 डेटा एक्सप्लोरर","⚖️ उपज भविष्यवाणी",
             "🌿 किस्म सलाहकार","🧪 खाद योजनाकार","🔮 पूर्व-कटाई",
             "🚨 जोखिम मॉनिटर","💬 AgriBot","🔢 मैनुअल भविष्यवाणी"],
    "block":"कृषि ब्लॉक","variety":"किस्म","hectares":"खेत का आकार (हे)",
    "soil":"मिट्टी का प्रकार","nursery":"नर्सरी","bundles":"गट्ठर गिनती",
    "dap":"DAP दिन 20 (किग्रा)","urea":"यूरिया दिन 40 (किग्रा)",
    "potash":"पोटाश दिन 50 (किग्रा)","micro":"सूक्ष्म पोषक दिन 70 (किग्रा)",
    "weed":"खरपतवार दिन 28 (किग्रा)","pest":"कीट नियंत्रण दिन 35 (मिली)",
    "actual_yield":"वास्तविक उपज (किग्रा)",
    "predict":"🔮 भविष्यवाणी करें","recommend":"🌿 सुझाव दें",
    "calc_dose":"🧪 खुराक गणना","estimate":"📦 अनुमान",
    "check_risk":"🚨 जोखिम जांचें",
    "predicted_yield":"अनुमानित उपज","recommended_variety":"अनुशंसित किस्म",
    "fertilizer_doses":"खाद खुराक","estimated_yield":"अनुमानित उपज",
    "risk_status":"जोखिम स्थिति",
    "excellent":"✅ उत्कृष्ट — औसत से ऊपर!",
    "average":"🟡 औसत — 15-20% सुधार संभव",
    "underperform":"⚠️ कम उत्पादन — अपेक्षा से नीचे!",
    "safe":"✅ सुरक्षित — अच्छा प्रदर्शन",
    "risky":"⚠️ जोखिम — कार्रवाई करें",
    "p1_desc":"45 खेत विशेषताओं से उपज की भविष्यवाणी",
    "p2_desc":"आपके ब्लॉक और मिट्टी के लिए सर्वोत्तम किस्म",
    "p3_desc":"6 खादों की इष्टतम खुराक की गणना",
    "p4_desc":"कटाई से पहले गट्ठर गिनकर उपज का अनुमान",
    "p5_desc":"कम प्रदर्शन करने वाले खेतों का पता लगाता है",
    "ask_me":"पूछें...","clear_chat":"🔄 साफ करें",
  },
  "తెలుగు":{
    "title":"AgriML India — స్మార్ట్ వ్యవసాయం",
    "pages":["🏠 డాష్‌బోర్డ్","📊 డేటా అన్వేషణ","⚖️ దిగుబడి అంచనా",
             "🌿 రకం సలహాదారు","🧪 ఎరువు ప్లానర్","🔮 ముందు-కోత",
             "🚨 రిస్క్ మానిటర్","💬 AgriBot","🔢 మాన్యువల్ అంచనా"],
    "block":"వ్యవసాయ బ్లాక్","variety":"రకం","hectares":"పొలం పరిమాణం (హె)",
    "soil":"మట్టి రకం","nursery":"నర్సరీ","bundles":"కట్టల సంఖ్య",
    "dap":"DAP రోజు 20 (కి)","urea":"యూరియా రోజు 40 (కి)",
    "potash":"పొటాష్ రోజు 50 (కి)","micro":"సూక్ష్మపోషకాలు రోజు 70 (కి)",
    "weed":"కలుపు రోజు 28 (కి)","pest":"తెగులు రోజు 35 (మి.లీ)",
    "actual_yield":"మీ నిజమైన దిగుబడి (కి)",
    "predict":"🔮 అంచనా వేయి","recommend":"🌿 సిఫారసు",
    "calc_dose":"🧪 మోతాదు లెక్కించు","estimate":"📦 అంచనా",
    "check_risk":"🚨 రిస్క్ తనిఖీ",
    "predicted_yield":"అంచనా దిగుబడి","recommended_variety":"సిఫారసు రకం",
    "fertilizer_doses":"ఎరువు మోతాదులు","estimated_yield":"అంచనా దిగుబడి",
    "risk_status":"రిస్క్ స్థితి",
    "excellent":"✅ అద్భుతం — సగటుకు మించి!",
    "average":"🟡 సగటు — 15-20% మెరుగు సాధ్యం",
    "underperform":"⚠️ తక్కువ దిగుబడి — సగటుకు తక్కువ!",
    "safe":"✅ సురక్షితం — మంచి పనితీరు",
    "risky":"⚠️ రిస్క్ — చర్య తీసుకో",
    "p1_desc":"45 పొలం అంశాల నుండి దిగుబడి అంచనా",
    "p2_desc":"మీ బ్లాక్ మరియు మట్టికి ఉత్తమ రకం",
    "p3_desc":"6 ఎరువులకు సరైన మోతాదులు",
    "p4_desc":"కోత ముందు కట్టలు లెక్కించి దిగుబడి అంచనా",
    "p5_desc":"తక్కువ పనితీరు పొలాలను గుర్తిస్తుంది",
    "ask_me":"అడగండి...","clear_chat":"🔄 క్లియర్",
  },
  "മലയാളം":{
    "title":"AgriML India — സ്മാർട്ട് കൃഷി",
    "pages":["🏠 ഡാഷ്‌ബോർഡ്","📊 ഡേറ്റ എക്‌സ്‌പ്ലോറർ","⚖️ വിളവ് പ്രവചനം",
             "🌿 ഇനം ഉപദേഷ്ടാവ്","🧪 വളം പ്ലാനർ","🔮 വിളവെടുപ്പ് മുൻ",
             "🚨 റിസ്ക് മോണിറ്റർ","💬 AgriBot","🔢 മാനുവൽ പ്രവചനം"],
    "block":"അഗ്രിബ്ലോക്ക്","variety":"ഇനം","hectares":"ഫാം വലിപ്പം (ഹെ)",
    "soil":"മണ്ണ് തരം","nursery":"നഴ്സറി","bundles":"കെട്ടുകളുടെ എണ്ണം",
    "dap":"DAP ദിവസം 20 (കി)","urea":"യൂറിയ ദിവസം 40 (കി)",
    "potash":"പൊട്ടാഷ് ദിവസം 50 (കി)","micro":"സൂക്ഷ്മ പോഷകം ദിവസം 70 (കി)",
    "weed":"കള നിയന്ത്രണം ദിവസം 28 (കി)","pest":"കീട നിയന്ത്രണം ദിവസം 35 (മി.ലി)",
    "actual_yield":"നിങ്ങളുടെ യഥാർത്ഥ വിളവ് (കി)",
    "predict":"🔮 പ്രവചിക്കുക","recommend":"🌿 ശുപാർശ",
    "calc_dose":"🧪 ഡോസ് കണക്കാക്കുക","estimate":"📦 കണക്കാക്കുക",
    "check_risk":"🚨 അപകടം പരിശോധിക്കുക",
    "predicted_yield":"പ്രവചിത വിളവ്","recommended_variety":"ശുപാർശ ചെയ്ത ഇനം",
    "fertilizer_doses":"വളം ഡോസുകൾ","estimated_yield":"കണക്കാക്കിയ വിളവ്",
    "risk_status":"അപകട നില",
    "excellent":"✅ മികച്ചത് — ശരാശരിക്ക് മുകളിൽ!",
    "average":"🟡 ശരാശരി — 15-20% മെച്ചപ്പെടൽ സാദ്ധ്യം",
    "underperform":"⚠️ കുറഞ്ഞ വിളവ് — പ്രതീക്ഷിതത്തിൽ താഴെ!",
    "safe":"✅ സുരക്ഷിതം — നല്ല പ്രകടനം",
    "risky":"⚠️ അപകടം — നടപടി എടുക്കൂ",
    "p1_desc":"45 ഫാം ഫീച്ചറുകളിൽ നിന്ന് വിളവ് പ്രവചനം",
    "p2_desc":"നിങ്ങളുടെ ബ്ലോക്കിനും മണ്ണിനും ഏറ്റവും നല്ല ഇനം",
    "p3_desc":"6 വളങ്ങൾക്ക് ശരിയായ ഡോസ് കണക്കാക്കുന്നു",
    "p4_desc":"കൊയ്ത്തിന് മുൻപ് കെട്ടുകൾ എണ്ണി വിളവ് കണക്കാക്കുക",
    "p5_desc":"മോശം പ്രകടനമുള്ള ഫാമുകൾ കണ്ടെത്തുന്നു",
    "ask_me":"ചോദിക്കൂ...","clear_chat":"🔄 മായ്ക്കുക",
  }
}

# ═══════════════════════════════════════════════════════════════
#  AGRIBOT DATA
# ═══════════════════════════════════════════════════════════════
BOT = {
  "variety":{
    "English":"🌾 **Best seed for your land:**\n👉 Cuddalore/Kallakurichi/Sankarapuram → **Co_43**\n👉 Chinnasalem/Panruti → **Delux Ponni**\n👉 Kurinjipadi → **Ponmani**\n\n✅ Use the Variety Advisor page for block-specific recommendation!",
    "தமிழ்":"🌾 **சிறந்த விதை:**\n👉 குடலூர்/சங்கரபுரம் → **Co_43**\n👉 சின்னசேலம் → **டீலக்ஸ் பொன்னி**\n👉 குரிஞ்சிபாடி → **பொன்மணி**",
    "हिंदी":"🌾 **बीज सुझाव:**\n👉 Cuddalore → **Co_43**\n👉 Chinnasalem → **Delux Ponni**\n👉 Kurinjipadi → **Ponmani**",
    "తెలుగు":"🌾 **విత్తన సూచన:**\n👉 Cuddalore → **Co_43**\n👉 Chinnasalem → **Delux Ponni**\n👉 Kurinjipadi → **Ponmani**",
    "മലയാളം":"🌾 **വിത്ത് ശുപാർശ:**\n👉 Cuddalore → **Co_43**\n👉 Chinnasalem → **Delux Ponni**\n👉 Kurinjipadi → **Ponmani**"
  },
  "fertilizer":{
    "English":"🧪 **Fertilizer schedule (per Ha):**\n📅 Day 20 → **40 Kg DAP** ← Most important!\n📅 Day 40 → **27 Kg Urea**\n📅 Day 50 → **10 Kg Potash**\n📅 Day 70 → **15 Kg Micronutrients**\n📅 Day 28 → **2 Kg Weed spray**\n📅 Day 35 → **600 ml Pest control**\n\n⚠️ Late DAP = 15-20% yield loss!",
    "தமிழ்":"🧪 **உர அட்டவணை:**\n📅 நாள் 20 → **40கி DAP** ← மிக முக்கியம்!\n📅 நாள் 40 → **27கி யூரியா**\n📅 நாள் 70 → **15கி நுண்ணூட்டம்**",
    "हिंदी":"🧪 **खाद समय-सारणी:**\n📅 दिन 20 → **40 किग्रा DAP** ← बहुत जरूरी!\n📅 दिन 40 → **27 किग्रा यूरिया**\n📅 दिन 70 → **15 किग्रा सूक्ष्म**",
    "తెలుగు":"🧪 **ఎరువు సమయపట్టిక:**\n📅 రోజు 20 → **40కి DAP** ← చాలా ముఖ్యం!\n📅 రోజు 40 → **27కి యూరియా**\n📅 రోజు 70 → **15కి సూక్ష్మపోషకాలు**",
    "മലയാളം":"🧪 **വളം ഷെഡ്യൂൾ:**\n📅 ദിവസം 20 → **40കി DAP** ← ഏറ്റവും പ്രധാനം!\n📅 ദിവസം 40 → **27കി യൂറിയ**\n📅 ദിവസം 70 → **15കി സൂക്ഷ്മ**"
  },
  "yield":{
    "English":"⚖️ **Expected yield by farm size:**\n🌾 1 Ha → ~5,757 Kg\n🌾 2 Ha → ~11,487 Kg\n🌾 3 Ha → ~17,266 Kg\n🌾 4 Ha → ~24,473 Kg\n🌾 5 Ha → ~31,046 Kg\n🌾 6 Ha → ~36,228 Kg",
    "தமிழ்":"⚖️ **ஹெக்டேர் வாரியாக மகசூல்:**\n🌾 1 → ~5,757கி\n🌾 3 → ~17,266கி\n🌾 5 → ~31,046கி",
    "हिंदी":"⚖️ **आकार के अनुसार उपज:**\n🌾 1 हे → ~5,757 किग्रा\n🌾 3 → ~17,266\n🌾 5 → ~31,046",
    "తెలుగు":"⚖️ **పరిమాణం వారీగా దిగుబడి:**\n🌾 1హె → ~5,757కి\n🌾 3 → ~17,266\n🌾 5 → ~31,046",
    "മലയാളം":"⚖️ **ഫാം വലിപ്പം → വിളവ്:**\n🌾 1ഹെ → ~5,757കി\n🌾 3 → ~17,266\n🌾 5 → ~31,046"
  },
  "improve":{
    "English":"💡 **Top ways to improve yield:**\n1️⃣ Apply DAP on Day 20 — never delay!\n2️⃣ Apply Urea on Day 40 exactly\n3️⃣ Weed spray on Day 28\n4️⃣ Micronutrients on Day 70\n5️⃣ Choose the right variety for your block\n\n🚀 All steps together = 15-20% more yield!",
    "தமிழ்":"💡 **மகசூல் அதிகரிக்க:**\n1️⃣ நாள் 20ல் DAP\n2️⃣ நாள் 40ல் யூரியா\n3️⃣ நாள் 28ல் களை\n4️⃣ நாள் 70ல் நுண்ணூட்டம்\n5️⃣ சரியான ரகம்",
    "हिंदी":"💡 **उपज बढ़ाने के तरीके:**\n1️⃣ दिन 20 पर DAP\n2️⃣ दिन 40 पर यूरिया\n3️⃣ दिन 28 पर खरपतवार\n4️⃣ दिन 70 पर सूक्ष्म\n5️⃣ सही किस्म",
    "తెలుగు":"💡 **దిగుబడి పెంచే మార్గాలు:**\n1️⃣ రోజు 20న DAP\n2️⃣ రోజు 40న యూరియా\n3️⃣ రోజు 28న కలుపు\n4️⃣ రోజు 70న సూక్ష్మం\n5️⃣ సరైన రకం",
    "మలయాళం":"💡 **വിളവ് മെച്ചപ്പെടുത്തൽ:**\n1️⃣ ദിവസം 20ന് DAP\n2️⃣ ദിവസം 40ന് യൂറിയ\n3️⃣ ദിവസം 28ന് കള\n4️⃣ ദിവസം 70ന് സൂക്ഷ്മം\n5️⃣ ശരിയായ ഇനം"
  },
  "soil":{
    "English":"🌱 **Soil guide:**\n🪨 Clay → Better water retention. Good for Co_43 and Delux Ponni. Drain field before harvest.\n🌿 Alluvial → Excellent for all varieties. Responds well to potash.",
    "தமிழ்":"🌱 **மண் வழிகாட்டி:**\n🪨 களிமண் → Co_43 சிறந்தது. அறுவடைக்கு முன் நீர்வடிப்பு.\n🌿 வண்டல் மண் → எல்லா ரகமும் நன்று.",
    "हिंदी":"🌱 **मिट्टी मार्गदर्शिका:**\n🪨 चिकनी → Co_43 सर्वोत्तम.\n🌿 जलोढ़ → सभी किस्में अच्छी.",
    "తెలుగు":"🌱 **మట్టి గైడ్:**\n🪨 బంకమట్టి → Co_43 ఉత్తమం.\n🌿 ఒండ్రు → అన్ని రకాలు మంచివి.",
    "മലയാളം":"🌱 **മണ്ണ് ഗൈഡ്:**\n🪨 കളിമണ്ണ് → Co_43 ഉത്തമം.\n🌿 എക്കൽ → എല്ലാ ഇനങ്ങളും നല്ലത്."
  },
  "risk":{
    "English":"🚨 **Farm risk summary:**\n⚠️ ~4.6% of farms underperform — producing 6,000+ Kg below similar farms.\n\n**Common causes:**\n❌ Late DAP application\n❌ Wrong variety for the block\n❌ Insufficient weed control\n❌ Skipped micronutrients\n\n✅ Use Risk Monitor to check your farm!",
    "தமிழ்":"🚨 **அபாய சுருக்கம்:**\n⚠️ 4.6% பண்ணைகள் குறைவான மகசூல்.\n❌ தாமதமான DAP முக்கிய காரணம்.",
    "हिंदी":"🚨 **जोखिम सारांश:**\n⚠️ 4.6% खेत कम उत्पादन.\n❌ मुख्य कारण: देर से DAP.",
    "తెలుగు":"🚨 **రిస్క్ సారాంశం:**\n⚠️ 4.6% పొలాలు తక్కువ దిగుబడి.\n❌ ప్రధాన కారణం: ఆలస్య DAP.",
    "മലയാളം":"🚨 **അപകട സംഗ്രഹം:**\n⚠️ 4.6% ഫാമുകൾ കുറഞ്ഞ വിളവ്.\n❌ DAP വൈകൽ പ്രധാന കാരണം."
  },
  "water":{
    "English":"💧 **Irrigation guide:**\n🌊 Keep 5 cm standing water during tillering.\n💦 Irrigate before and after each fertilizer application.\n🚿 Drain field 10-12 days before harvest.",
    "தமிழ்":"💧 **நீர்ப்பாசன வழிகாட்டி:**\n🌊 நாற்று நட்ட நேரம் 5 செமீ நீர்.\n🚿 அறுவடைக்கு 10 நாள் முன் நீர்வடிக்கவும்.",
    "हिंदी":"💧 **सिंचाई मार्गदर्शिका:**\n🌊 रोपाई में 5 सेमी पानी रखें.\n🚿 कटाई से 10 दिन पहले पानी निकालें.",
    "తెలుగు":"💧 **నీటిపారుదల:**\n🌊 5 సెమీ నీరు ఉంచండి.\n🚿 కోత కు 10 రోజుల ముందు నీరు తీయండి.",
    "മലയാളം":"💧 **ജലസേചന ഗൈഡ്:**\n🌊 5 സെ.മി. വെള്ളം ഉറപ്പ് വരുത്തുക.\n🚿 കൊയ്ത്തിന് 10 ദിവസം മുൻപ് വെള്ളം ഒഴുക്കുക."
  },
  "preharvest":{
    "English":"🔮 **Bundle count → yield estimate:**\n📦 80-100 bundles → ~5,800 Kg\n📦 200-260 → ~15,800 Kg\n📦 300-360 → ~21,000 Kg\n📦 480-540 → ~35,000 Kg\n\n✅ R²=0.99 — nearly perfect on 2,338 real farms!",
    "தமிழ்":"🔮 **கட்டுகள் → மகசூல்:**\n📦 80-100 → ~5,800கி\n📦 300-360 → ~21,000கி\n📦 480-540 → ~35,000கி",
    "हिंदी":"🔮 **गट्ठर → उपज:**\n📦 80-100 → ~5,800 किग्रा\n📦 300-360 → ~21,000\n📦 480-540 → ~35,000",
    "తెలుగు":"🔮 **కట్టలు → దిగుబడి:**\n📦 80-100 → ~5,800కి\n📦 300-360 → ~21,000\n📦 480-540 → ~35,000",
    "മലയാളം":"🔮 **കെട്ടുകൾ → വിളവ്:**\n📦 80-100 → ~5,800കി\n📦 300-360 → ~21,000\n📦 480-540 → ~35,000"
  }
}

GREET = {
  "English":"👋 **Hello Farmer!** I'm AgriBot 🌾\nTrained on 2,338 Tamil Nadu paddy farms.\nAsk me about seeds, fertilizer, yield, risk, or soil!",
  "தமிழ்":"👋 **வணக்கம்!** நான் AgriBot 🌾\n2,338 பண்ணைகள் தரவில் பயிற்சி.\nவிதை, உரம், மகசூல் பற்றி கேளுங்கள்!",
  "हिंदी":"👋 **नमस्ते!** मैं AgriBot हूँ 🌾\n2,338 खेतों पर ट्रेन.\nबीज, खाद, उपज के बारे में पूछें!",
  "తెలుగు":"👋 **నమస్కారం!** నేను AgriBot 🌾\n2,338 పొలాల డేటా.\nవిత్తనం, ఎరువు, దిగుబడి గురించి అడగండి!",
  "മലയാളം":"👋 **നമസ്കാരം!** ഞാൻ AgriBot 🌾\n2,338 ഫാം ഡേറ്റ.\nവിത്ത്, വളം, വിളവ് കുറിച്ച് ചോദിക്കൂ!"
}

def detect_intent(text):
    t = text.lower()
    if any(w in t for w in ["fertil","urea","dap","potash","dose","manure","micro",
        "உரம","खाद","ఎరువు","വളം"]): return "fertilizer"
    if any(w in t for w in ["yield","kg","kilo","output","மகசூல்","उपज","దిగుబడి","വിളവ്"]): return "yield"
    if any(w in t for w in ["improv","increas","boost","more","better","மேம்","अधिक","పెంచ","മെച്ച"]): return "improve"
    if any(w in t for w in ["soil","clay","alluvial","land","மண்","मिट्टी","మట్టి","മണ്ണ്"]): return "soil"
    if any(w in t for w in ["risk","danger","underperform","problem","ஆபத்","खतरा","రిస్క్","അപകടം"]): return "risk"
    if any(w in t for w in ["water","irrig","rain","canal","borewell","நீர்","पानी","నీరు","വെള്ളം"]): return "water"
    if any(w in t for w in ["bundle","trash","before harvest","pre","count","கட்டு","गट्ठर","కట్ట","കെട്ട്"]): return "preharvest"
    return "variety"

# ═══════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════
def page_header(subtitle, title, desc):
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a3d12,#2d6a1f);
border-radius:16px;padding:24px 28px;margin-bottom:20px;color:#fff;">
  <div style="font-size:11px;font-weight:700;letter-spacing:3px;
  color:#a5d6a7;text-transform:uppercase;margin-bottom:4px;">{subtitle}</div>
  <div style="font-family:'Montserrat',sans-serif;font-size:26px;
  font-weight:900;margin-bottom:6px;">{title}</div>
  <div style="font-size:13px;color:rgba(255,255,255,0.75);">{desc}</div>
</div>""", unsafe_allow_html=True)

def card(title, value, subtitle="", color="#2d6a1f"):
    st.markdown(f"""
<div style="background:#fff;border:1px solid #d6e8d6;border-left:4px solid {color};
border-radius:12px;padding:14px 16px;margin-bottom:10px;">
  <div style="font-size:11px;color:#5a7a5a;font-weight:700;text-transform:uppercase;
  letter-spacing:1px;">{title}</div>
  <div style="font-size:24px;font-weight:900;color:{color};margin:4px 0;">{value}</div>
  <div style="font-size:12px;color:#888;">{subtitle}</div>
</div>""", unsafe_allow_html=True)

def result_box(value, label, detail=""):
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#e8f5e9,#f1f8f0);
border:2px solid #4caf50;border-radius:16px;padding:20px 24px;
text-align:center;margin:16px 0;">
  <div style="font-family:'Montserrat',sans-serif;font-size:36px;
  font-weight:900;color:#1a3d12;">{value}</div>
  <div style="font-size:13px;font-weight:700;color:#2d6a1f;
  letter-spacing:2px;text-transform:uppercase;">{label}</div>
  <div style="font-size:12px;color:#5a7a5a;margin-top:4px;">{detail}</div>
</div>""", unsafe_allow_html=True)

def solution_box(icon, title, body, color="#2d6a1f"):
    st.markdown(f"""
<div style="background:#fff;border:1px solid #d6e8d6;border-left:4px solid {color};
border-radius:12px;padding:12px 16px;margin-bottom:10px;">
  <div style="font-size:13px;font-weight:800;color:{color};">{icon} {title}</div>
  <div style="font-size:12.5px;color:#3a4a3a;margin-top:4px;">{body}</div>
</div>""", unsafe_allow_html=True)

def lc(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a3d12")),
        paper_bgcolor="#ffffff", plot_bgcolor="#f7fbf7",
        font=dict(color="#3a4a3a"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="#f7fbf7"),
    )
    fig.update_xaxes(gridcolor="#d6e8d6")
    fig.update_yaxes(gridcolor="#d6e8d6")

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="padding:16px 12px 10px;">
  <div style="font-family:'Montserrat',sans-serif;font-size:20px;
  font-weight:900;color:#fff;letter-spacing:2px;">
    AGRI<span style="color:#a5d6a7;">ML</span>
    <span style="color:rgba(255,255,255,0.45);font-size:10px;"> INDIA</span>
  </div>
  <div style="font-size:9px;color:rgba(255,255,255,0.45);
  letter-spacing:3px;font-weight:700;">SMART FARMING · v6.0</div>
  <div style="height:1px;background:rgba(255,255,255,0.15);margin:10px 0;"></div>
</div>""", unsafe_allow_html=True)

    if "app_lang" not in st.session_state:
        st.session_state.app_lang = "English"

    lang = st.selectbox("🌐 Language",LANGS,
        index=LANGS.index(st.session_state.app_lang),key="lang_sel")
    st.session_state.app_lang = lang
    L = T[lang]

    page = st.radio("", L["pages"], label_visibility="collapsed")

    st.markdown(f"""
<div style="margin:14px 4px 0;background:rgba(255,255,255,0.10);
border-radius:12px;padding:12px;font-size:10.5px;
color:rgba(255,255,255,0.80);line-height:2.0;
border:1px solid rgba(255,255,255,0.18);">
  ✅ <b style="color:#a5d6a7;">2,338 farms</b> analyzed<br/>
  📍 6 Agriblocks · 3 Varieties<br/>
  🎯 45 Features · Tamil Nadu<br/>
  🤖 5 ML Models · R²=0.99
</div>""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────
M_ALL = train_all_models()

if M_ALL is None:
    st.error("⚠️ paddydataset.csv not found. Please place it in the same folder as this app.")
    st.stop()

df_raw = M_ALL["df"]
PNAMES = L["pages"]

# ═══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════
if page == PNAMES[0]:
    page_header("Tamil Nadu Paddy Analytics", L["title"],
                "2,338 farms · 6 Agriblocks · 45 Features · 5 ML Models")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Farms","2,338","After cleaning")
    c2.metric("Yield R²","0.9894","P1 model")
    c3.metric("Variety Acc","99.0%","P2 model")
    c4.metric("Fertilizer Acc","100%","P3 model")
    c5.metric("Anomalies",f"{M_ALL['p5']['anomaly_count']}","4.6% flagged")

    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1:
        avg = df_raw.groupby("Agriblock")["Paddy yield(in Kg)"].mean().sort_values()
        fig = go.Figure(go.Bar(y=avg.index, x=avg.values, orientation="h",
            marker=dict(color=G, line=dict(width=0)),
            text=[f"{v:,.0f}" for v in avg.values], textposition="outside"))
        lc(fig,"📊 Avg Yield by Agriblock"); fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with c2:
        vc = df_raw["Variety"].value_counts()
        fig = go.Figure(go.Pie(labels=vc.index, values=vc.values,
            marker=dict(colors=["#2d6a1f","#e65100","#1565c0"],
                        line=dict(width=0)), hole=0.55,
            textinfo="label+percent"))
        lc(fig,"🥧 Variety Distribution")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with c3:
        hav = sorted(df_raw["Hectares"].unique())
        avgy = [df_raw[df_raw["Hectares"]==h]["Paddy yield(in Kg)"].mean() for h in hav]
        fig = go.Figure(go.Bar(x=hav, y=avgy, marker=dict(color=G, line=dict(width=0)),
            text=[f"{v:,.0f}" for v in avgy], textposition="outside"))
        lc(fig,"📈 Avg Yield by Farm Size (Ha)")
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="Hectares")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")
    st.markdown("### 🔍 Key Insights")
    c1,c2 = st.columns(2)
    with c1:
        solution_box("🏆","Sankarapuram — Most Efficient Block",
            "Highest average yield per hectare. Co_43 + alluvial soil combination.", "#2d6a1f")
        solution_box("💉","DAP on Day 20 = #1 Factor",
            "Top feature in every ML model. Never delay DAP application!", "#e65100")
        solution_box("📦","Trash Bundles = 99% Accurate",
            "Just count bundles before harvest — R²=0.99 on real farm data.", "#1565c0")
    with c2:
        solution_box("🌾","Co_43 Dominates",
            "Best variety in 3 out of 6 blocks. Delux Ponni wins in 2 blocks.", "#2d6a1f")
        solution_box("⚠️",f"{M_ALL['p5']['anomaly_count']} Underperforming Farms",
            f"4.6% of farms produce significantly below similar farms. Check Risk Monitor.", "#c62828")
        solution_box("📐","Larger Farms More Efficient",
            "6 Ha farms are ~7% more efficient per hectare than 1 Ha farms.", "#6a1b9a")

# ═══════════════════════════════════════════════════════════════
#  PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════
elif page == PNAMES[1]:
    page_header("Raw Data + Statistics", "📊 Data Explorer",
                f"Explore {len(df_raw):,} farm records · 45 features")

    t1,t2,t3 = st.tabs(["📋 Dataset","📊 Charts","🔢 Statistics"])
    with t1:
        st.dataframe(df_raw, use_container_width=True, height=400)
        csv = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "paddydataset_clean.csv", "text/csv")
    with t2:
        col1,col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Histogram(
                x=df_raw["Paddy yield(in Kg)"], nbinsx=30,
                marker_color="rgba(45,106,31,0.65)", marker_line_width=0))
            lc(fig,"📈 Yield Distribution")
            fig.update_xaxes(title_text="Yield (Kg)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        with col2:
            fig = go.Figure(go.Scatter(
                x=df_raw["Trash(in bundles)"], y=df_raw["Paddy yield(in Kg)"],
                mode="markers", marker=dict(color="rgba(45,106,31,0.3)",size=5)))
            lc(fig,"📦 Trash Bundles vs Yield")
            fig.update_xaxes(title_text="Trash Bundles")
            fig.update_yaxes(title_text="Yield (Kg)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with t3:
        st.dataframe(df_raw.describe().round(2), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE: YIELD PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif page == PNAMES[2]:
    page_header("Exact Kg Prediction", "⚖️ " + L["predicted_yield"],
                L["p1_desc"])
    m1 = M_ALL["p1"]
    c1,c2,c3 = st.columns(3)
    c1.metric("Test MAE",f"±{m1['mae']:.0f} Kg","Test set")
    c2.metric("Test R²",f"{m1['r2']:.4f}","Near perfect")
    c3.metric("CV MAE",f"±{m1['cv_mae']:.0f} Kg",f"5-fold ±{m1['cv_std']:.0f}")
    st.markdown("---")

    # Feature importance chart
    rf = m1["model"]
    feats = m1["features"]
    imp_df = pd.DataFrame({"Feature":feats,"Imp":rf.feature_importances_}
                          ).sort_values("Imp",ascending=False).head(12)
    fig = go.Figure(go.Bar(y=imp_df["Feature"], x=imp_df["Imp"],
        orientation="h", marker=dict(color=G*3, line=dict(width=0))))
    lc(fig,"🔍 Top 12 Features for Yield Prediction")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")
    st.markdown(f"#### {L['predict']} — Enter Your Farm Details")
    c1,c2,c3,c4 = st.columns(4)
    p1_blk  = c1.selectbox(L["block"],  BL, key="p1blk")
    p1_var  = c2.selectbox(L["variety"], VA, key="p1var")
    p1_ha   = c3.selectbox(L["hectares"],[1,2,3,4,5,6],index=2,key="p1ha")
    p1_soil = c4.selectbox(L["soil"],    ST, key="p1soil")
    c5,c6 = st.columns(2)
    p1_nurs = c5.selectbox(L["nursery"], NU, key="p1nurs")
    p1_seed = c6.number_input("Seedrate (Kg)",5,200,150,key="p1seed")

    st.markdown(f"##### 💊 Fertilizer Schedule")
    c1,c2,c3 = st.columns(3)
    p1_dap  = c1.number_input(L["dap"],   0,400,int(p1_ha*40), key="p1dap")
    p1_urea = c2.number_input(L["urea"],  0,400,int(p1_ha*27), key="p1urea")
    p1_pot  = c3.number_input(L["potash"],0,200,int(p1_ha*10), key="p1pot")
    c4,c5,c6 = st.columns(3)
    p1_micro= c4.number_input(L["micro"], 0,200,int(p1_ha*15), key="p1micro")
    p1_weed = c5.number_input(L["weed"],  0,50, int(p1_ha*2),  key="p1weed")
    p1_pest = c6.number_input(L["pest"],  0,5000,int(p1_ha*600),key="p1pest")
    p1_bun  = st.slider(L["bundles"],80,600,int(p1_ha*85),key="p1bun")

    if st.button(L["predict"], key="p1btn", use_container_width=True):
        le_dict = M_ALL["le_dict"]
        df_enc  = M_ALL["df_enc"]
        avg_row = df_enc[df_enc["Hectares"]==p1_ha].mean(numeric_only=True)
        inp = {col: avg_row[col] if col in avg_row else 0 for col in feats}
        inp["Hectares"]     = p1_ha
        inp["DAP_20days"]   = p1_dap
        inp["Urea_40Days"]  = p1_urea
        inp["Potassh_50Days"] = p1_pot
        inp["Micronutrients_70Days"] = p1_micro
        inp["Weed28D_thiobencarb"]   = p1_weed
        inp["Pest_60Day(in ml)"]     = p1_pest
        inp["Trash(in bundles)"]     = p1_bun
        inp["Seedrate(in Kg)"]       = p1_seed
        for col, le in le_dict.items():
            if col == "Agriblock" and col in feats:
                inp[col] = le.transform([p1_blk])[0]
            elif col == "Variety" and col in feats:
                inp[col] = le.transform([p1_var])[0]
            elif col == "Soil Types" and col in feats:
                inp[col] = le.transform([p1_soil])[0]
            elif col == "Nursery" and col in feats:
                inp[col] = le.transform([p1_nurs])[0]
        X_inp = pd.DataFrame([inp])[feats]
        pred_yield = int(rf.predict(X_inp)[0])
        expected   = YH[p1_ha]
        gap        = pred_yield - expected

        result_box(f"{pred_yield:,}", L["predicted_yield"],
                   f"{p1_blk} · {p1_var} · {p1_ha} Ha · MAE ±{m1['mae']:.0f} Kg")

        if gap >= 1000:
            st.success(L["excellent"])
        elif gap >= -500:
            st.warning(L["average"])
        else:
            st.error(L["underperform"])

        best_v = BV[p1_blk]
        if p1_var != best_v:
            solution_box("🌿",f"Switch to {best_v}",
                f"{best_v} is the best-performing variety in {p1_blk}. "
                f"Switching can add 300-600 Kg/Ha.", "#1565c0")
        if p1_dap < p1_ha*36:
            solution_box("🔴","DAP Under-dosed",
                f"You applied {p1_dap} Kg. Optimal: {int(p1_ha*40)} Kg on Day 20. "
                f"Late/low DAP = 15-20% yield loss!", "#c62828")
        else:
            solution_box("✅","DAP Dose Correct",
                f"{p1_dap} Kg applied — on target for {p1_ha} Ha.", "#2d6a1f")

# ═══════════════════════════════════════════════════════════════
#  PAGE: VARIETY ADVISOR
# ═══════════════════════════════════════════════════════════════
elif page == PNAMES[3]:
    page_header("Best Variety Per Block", "🌿 " + L["recommended_variety"],
                L["p2_desc"])
    m2 = M_ALL["p2"]
    c1,c2 = st.columns(2)
    c1.metric("Test Accuracy",f"{m2['acc']:.4f}","Classification")
    c2.metric("CV Accuracy",f"{m2['cv_acc']:.4f}",f"5-fold ±{m2['cv_std']:.4f}")

    # Block-variety chart
    by = {}
    for v in VA:
        by[v] = df_raw[df_raw["Variety"]==v].groupby("Agriblock")["Paddy yield(in Kg)"].mean()
    fig = go.Figure()
    colors_v = ["#2d6a1f","#e65100","#1565c0"]
    for (var,d),col in zip(by.items(),colors_v):
        fig.add_trace(go.Bar(name=var, x=d.index, y=d.values,
            marker=dict(color=col,line=dict(width=0))))
    lc(fig,"📊 Yield by Block & Variety")
    fig.update_layout(barmode="group")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")
    st.markdown(f"#### {L['recommend']}")
    c1,c2,c3 = st.columns(3)
    v2_blk  = c1.selectbox(L["block"], BL, key="v2blk")
    v2_soil = c2.selectbox(L["soil"],  ST, key="v2soil")
    v2_ha   = c3.selectbox(L["hectares"],[1,2,3,4,5,6],index=2,key="v2ha")

    if st.button(L["recommend"], key="v2btn", use_container_width=True):
        le_dict = M_ALL["le_dict"]
        df_enc  = M_ALL["df_enc"]
        feat2   = m2["features"]
        avg2    = df_enc[df_enc["Hectares"]==v2_ha].mean(numeric_only=True)
        inp2 = {col: avg2[col] if col in avg2 else 0 for col in feat2}
        inp2["Hectares"] = v2_ha
        for col, le in le_dict.items():
            if col == "Agriblock" and col in feat2:
                inp2[col] = le.transform([v2_blk])[0]
            elif col == "Soil Types" and col in feat2:
                inp2[col] = le.transform([v2_soil])[0]
        X2_inp = pd.DataFrame([inp2])[feat2]
        pred_code = m2["model"].predict(X2_inp)[0]
        pred_var  = m2["le"].inverse_transform([pred_code])[0]

        result_box(pred_var, L["recommended_variety"],
                   f"{v2_blk} · {v2_soil} soil · {v2_ha} Ha · Accuracy {m2['acc']:.1%}")
        solution_box("✅",f"{pred_var} is best for {v2_blk}",
            "Use certified seeds only. Consistently outperforms other varieties in this block.",
            "#2d6a1f")
        best_known = BV[v2_blk]
        if pred_var == best_known:
            solution_box("📊","Confirmed by historical data",
                f"Historical farm data also shows {pred_var} leads in {v2_blk}.", "#1565c0")

# ═══════════════════════════════════════════════════════════════
#  PAGE: FERTILIZER PLANNER
# ═══════════════════════════════════════════════════════════════
elif page == PNAMES[4]:
    page_header("Dose Optimizer", "🧪 " + L["fertilizer_doses"],
                L["p3_desc"])
    m3 = M_ALL["p3"]
    st.metric("Model Accuracy",f"{m3['mean_acc']:.4f}",
              "100% — doses are deterministic function of farm size")

    # Dose chart
    hav = [1,2,3,4,5,6]
    fig = go.Figure()
    for nm,rate,col in [("DAP",40,"#2d6a1f"),("Urea",27.13,"#e65100"),
                         ("Potash",10.38,"#1565c0"),("Micro",15,"#6a1b9a")]:
        fig.add_trace(go.Scatter(x=hav, y=[rate*h for h in hav], name=nm,
            mode="lines+markers", line=dict(color=col,width=2.5),
            marker=dict(size=7)))
    lc(fig,"📈 Fertilizer Dose vs Farm Size")
    fig.update_xaxes(title_text="Farm Size (Ha)")
    fig.update_yaxes(title_text="Dose (Kg)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")
    st.markdown(f"#### {L['calc_dose']}")
    c1,c2,c3 = st.columns(3)
    f3_ha   = c1.selectbox(L["hectares"],[1,2,3,4,5,6],index=2,key="f3ha")
    f3_blk  = c2.selectbox(L["block"],   BL, key="f3blk")
    f3_soil = c3.selectbox(L["soil"],    ST, key="f3soil")

    if st.button(L["calc_dose"], key="f3btn", use_container_width=True):
        le_dict = M_ALL["le_dict"]
        df_enc  = M_ALL["df_enc"]
        feat3   = m3["features"]
        avg3    = df_enc[df_enc["Hectares"]==f3_ha].mean(numeric_only=True)
        inp3 = {col: avg3[col] if col in avg3 else 0 for col in feat3}
        inp3["Hectares"] = f3_ha
        for col, le in le_dict.items():
            if col == "Agriblock" and col in feat3:
                inp3[col] = le.transform([f3_blk])[0]
            elif col == "Soil Types" and col in feat3:
                inp3[col] = le.transform([f3_soil])[0]
        X3_inp = pd.DataFrame([inp3])[feat3]

        st.markdown(f"""
<div style="background:linear-gradient(135deg,#e8f5e9,#f1f8f0);
border:2px solid #a5c8a5;border-radius:16px;padding:20px 24px;margin:12px 0;">
<div style="font-size:15px;font-weight:800;color:#1a3d12;margin-bottom:14px;">
🧪 {L['fertilizer_doses']} — {f3_ha} Ha · {f3_blk}</div>""",
            unsafe_allow_html=True)

        doses = {}
        units = {"DAP_20days":"Kg","Weed28D_thiobencarb":"Kg","Urea_40Days":"Kg",
                 "Potassh_50Days":"Kg","Micronutrients_70Days":"Kg","Pest_60Day(in ml)":"ml"}
        cols_d = st.columns(3)
        icons = ["🌱","🌿","⚗️","⚗️","🍃","🐛"]
        for i,(target, mdl) in enumerate(m3["models"].items()):
            rec = mdl.predict(X3_inp)[0]
            doses[target] = rec
            u = units.get(target,"Kg")
            cols_d[i%3].metric(f"{icons[i]} {target.split('(')[0]}", f"{rec} {u}")

        st.markdown("</div>",unsafe_allow_html=True)
        solution_box("⚠️",f"Most Critical: Apply DAP on Day 20",
            f"Recommended: {doses.get('DAP_20days','40')} Kg for {f3_ha} Ha. "
            "Late application = 15-20% yield loss!", "#c62828")
        if f3_soil == "alluvial":
            solution_box("🌿","Alluvial Soil Tip",
                "Alluvial soil responds well to potash. Ensure proper irrigation.", "#1565c0")

# ═══════════════════════════════════════════════════════════════
#  PAGE: PRE-HARVEST
# ═══════════════════════════════════════════════════════════════
elif page == PNAMES[5]:
    page_header("Estimate Before Harvest","🔮 " + L["estimated_yield"],
                L["p4_desc"])
    m4 = M_ALL["p4"]
    c1,c2,c3 = st.columns(3)
    c1.metric("Test MAE",f"±{m4['mae']:.0f} Kg")
    c2.metric("Test R²",f"{m4['r2']:.4f}")
    c3.metric("CV MAE",f"±{m4['cv_mae']:.0f} Kg",f"5-fold ±{m4['cv_std']:.0f}")

    # Scatter chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_raw["Trash(in bundles)"],
        y=df_raw["Paddy yield(in Kg)"], mode="markers",
        marker=dict(color="rgba(45,106,31,0.3)",size=5), name="Farms"))
    m_,b_ = np.polyfit(df_raw["Trash(in bundles)"],df_raw["Paddy yield(in Kg)"],1)
    xs = np.linspace(df_raw["Trash(in bundles)"].min(),
                     df_raw["Trash(in bundles)"].max(),100)
    fig.add_trace(go.Scatter(x=xs, y=m_*xs+b_, mode="lines",
        line=dict(color="#e65100",width=2,dash="dot"), name="Trend"))
    lc(fig,"📈 Trash Bundles vs Yield (R=0.96)")
    fig.update_xaxes(title_text="Trash Bundles")
    fig.update_yaxes(title_text="Yield (Kg)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")
    c1,c2 = st.columns(2)
    ph_bun = c1.slider(L["bundles"],80,600,300,key="ph_bun")
    ph_ha  = c2.selectbox(L["hectares"],[1,2,3,4,5,6],index=2,key="ph_ha")

    live = int(m4["model"].predict(
        pd.DataFrame({"Trash(in bundles)":[ph_bun],"Hectares":[ph_ha]}))[0])
    st.markdown(f"""
<div style="background:#e8f5e9;border:1px solid #a5c8a5;border-radius:10px;
padding:10px 16px;font-size:13px;font-weight:700;color:#1a3d12;">
📦 Live Estimate: ~{live:,} Kg
</div>""", unsafe_allow_html=True)

    if st.button(L["estimate"], key="ph_btn", use_container_width=True):
        result_box(f"{live:,}", L["estimated_yield"],
                   f"{ph_bun} bundles · {ph_ha} Ha · MAE ±{m4['mae']:.0f} Kg")
        exp4 = YH[ph_ha]
        if live >= exp4:
            st.success(L["excellent"])
            solution_box("✅","On Track",
                f"Expected {exp4:,} Kg for {ph_ha} Ha. Your estimate {live:,} Kg is above target!",
                "#2d6a1f")
        else:
            st.warning(L["average"])
            solution_box("⚠️","Below Expected",
                f"Expected {exp4:,} Kg but estimated {live:,} Kg. "
                f"Gap: {exp4-live:,} Kg. Check fertilizer schedule.", "#e65100")
        solution_box("🌾","Harvest Timing",
            "Harvest when 80% grains are golden. Ideal moisture: 20-22%.", "#e65100")

# ═══════════════════════════════════════════════════════════════
#  PAGE: RISK MONITOR
# ═══════════════════════════════════════════════════════════════
elif page == PNAMES[6]:
    page_header("Farm Risk Detection","🚨 " + L["risk_status"],
                L["p5_desc"])
    m5 = M_ALL["p5"]
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Farms",f"{m5['total']:,}")
    c2.metric("Anomalies",f"{m5['anomaly_count']}",
              f"{m5['anomaly_count']/m5['total']*100:.1f}%")
    c3.metric("Normal Farms",f"{m5['total']-m5['anomaly_count']}")

    # PCA scatter
    vis = m5["df_vis"]
    fig = go.Figure()
    for lbl, col in [("Normal","steelblue"),("Anomaly","crimson")]:
        grp = vis[vis["Label"]==lbl]
        fig.add_trace(go.Scatter(x=grp["PC1"], y=grp["PC2"],
            mode="markers", name=lbl,
            marker=dict(color=col, size=5, opacity=0.6)))
    lc(fig,"🔴 Farm Anomaly Detection — PCA View")
    fig.update_xaxes(title_text="PC 1"); fig.update_yaxes(title_text="PC 2")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # Anomalies per block
    block_a = (vis[vis["Label"]=="Anomaly"].groupby("Agriblock")
               .size().reset_index().rename(columns={0:"Count"})
               .sort_values("Count",ascending=False))
    fig2 = go.Figure(go.Bar(x=block_a["Agriblock"], y=block_a["Count"],
        marker=dict(color=["#c62828","#e53935","#ef5350","#ef9a9a","#ffcdd2","#fff"],
                    line=dict(width=0))))
    lc(fig2,"📊 Anomalies per Agriblock")
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")
    st.markdown(f"#### {L['check_risk']}")
    c1,c2,c3 = st.columns(3)
    r5_blk  = c1.selectbox(L["block"],  BL, key="r5blk")
    r5_ha   = c2.selectbox(L["hectares"],[1,2,3,4,5,6],index=2,key="r5ha")
    r5_yld  = c3.number_input(L["actual_yield"],1000,50000,18000,key="r5yld")

    if st.button(L["check_risk"], key="r5btn", use_container_width=True):
        exp5 = YH[r5_ha]
        gap5 = r5_yld - exp5
        if gap5 < -1500:
            st.error(f"⚠️ {L['risky']}  —  Gap: {gap5:,} Kg vs expected {exp5:,} Kg")
            solution_box("🔧","Immediate Action Needed",
                f"Farm is {abs(gap5):,} Kg below expected. Check: DAP applied Day 20? "
                f"Right variety for {r5_blk}? Weed control done?","#c62828")
        elif gap5 < 0:
            st.warning(f"🟡 {L['average']}  —  Gap: {gap5:,} Kg")
            solution_box("📈","Improvement Possible",
                "Small improvements in fertilizer timing can close this gap.","#e65100")
        else:
            st.success(f"✅ {L['safe']}  —  +{gap5:,} Kg above expected!")
            solution_box("🏆","Keep Up The Same Schedule",
                f"Maintain fertilizer schedule and variety for {r5_blk} next season.","#2d6a1f")

# ═══════════════════════════════════════════════════════════════
#  PAGE: AGRIBOT
# ═══════════════════════════════════════════════════════════════
elif page == PNAMES[7]:
    page_header("AI Farming Assistant","💬 AgriBot",
                "Ask anything about seeds, fertilizer, yield, risk — in any language")

    if "chat_hist" not in st.session_state or \
       st.session_state.get("chat_lang") != lang:
        st.session_state.chat_hist = [{"role":"bot","text":GREET[lang]}]
        st.session_state.chat_lang = lang

    # Quick chips
    chips = {
        "English":["🌱 Best seed?","💊 Fertilizer?","📦 Yield?","📈 Improve","🚨 Risk","💧 Water"],
        "தமிழ்":["🌱 விதை?","💊 உரம்?","📦 மகசூல்?","📈 மேம்பாடு","🚨 ஆபத்து","💧 நீர்"],
        "हिंदी":["🌱 बीज?","💊 खाद?","📦 उपज?","📈 सुधार","🚨 जोखिम","💧 पानी"],
        "తెలుగు":["🌱 విత్తనం?","💊 ఎరువు?","📦 దిగుబడి?","📈 మెరుగు","🚨 రిస్క్","💧 నీరు"],
        "മലയാളം":["🌱 വിത്ത്?","💊 വളം?","📦 വിളവ്?","📈 മെച്ചം","🚨 അപകടം","💧 ജലം"]
    }
    intents = ["variety","fertilizer","yield","improve","risk","water"]

    cc = st.columns(6)
    for i,(col,chip,intent) in enumerate(zip(cc,chips[lang],intents)):
        with col:
            if st.button(chip,key=f"chip_{i}",use_container_width=True):
                st.session_state.chat_hist.append({"role":"user","text":chip})
                resp = BOT[intent].get(lang, BOT[intent]["English"])
                st.session_state.chat_hist.append({"role":"bot","text":resp})
                st.rerun()

    st.markdown("---")
    chat_html = ""
    for msg in st.session_state.chat_hist[-16:]:
        txt = msg["text"].replace("\n","<br/>")
        if msg["role"] == "bot":
            chat_html += f'''<div style="display:flex;gap:10px;margin-bottom:14px;">
<div style="width:34px;height:34px;border-radius:50%;
background:linear-gradient(135deg,#2d6a1f,#3d8a28);
display:flex;align-items:center;justify-content:center;
font-size:16px;flex-shrink:0;">🤖</div>
<div style="background:#fff;border:1px solid #d6e8d6;
border-radius:4px 16px 16px 16px;padding:12px 15px;
font-size:12.5px;line-height:1.7;color:#2e3d2e;
max-width:80%;box-shadow:0 2px 8px rgba(45,106,31,0.07);">{txt}</div>
</div>'''
        else:
            chat_html += f'''<div style="display:flex;gap:10px;
flex-direction:row-reverse;margin-bottom:14px;">
<div style="width:34px;height:34px;border-radius:50%;
background:linear-gradient(135deg,#e65100,#bf360c);
display:flex;align-items:center;justify-content:center;
font-size:16px;flex-shrink:0;">👨‍🌾</div>
<div style="background:linear-gradient(135deg,#2d6a1f,#3d8a28);
border-radius:16px 4px 16px 16px;padding:12px 15px;
font-size:12.5px;line-height:1.7;color:#fff;max-width:80%;">{txt}</div>
</div>'''

    st.markdown(f'''<div style="background:#f7fbf7;border:1px solid #d6e8d6;
border-radius:16px;padding:18px;height:360px;overflow-y:auto;
box-shadow:inset 0 2px 8px rgba(0,0,0,0.03);margin-bottom:12px;">
{chat_html}</div>''', unsafe_allow_html=True)

    placeholder = {"English":"Ask me anything...","தமிழ்":"கேளுங்கள்...",
                   "हिंदी":"पूछें...","తెలుగు":"అడగండి...","മലയാളം":"ചോദിക്കൂ..."}
    user_in = st.chat_input(placeholder.get(lang,"Ask me anything..."))
    if user_in:
        intent = detect_intent(user_in)
        resp   = BOT.get(intent,BOT["variety"]).get(lang,BOT.get(intent,BOT["variety"])["English"])
        st.session_state.chat_hist.append({"role":"user","text":user_in})
        st.session_state.chat_hist.append({"role":"bot","text":resp})
        st.rerun()

    if st.button(L["clear_chat"], key="clr"):
        st.session_state.chat_hist = [{"role":"bot","text":GREET[lang]}]
        st.rerun()

# ═══════════════════════════════════════════════════════════════
#  PAGE: MANUAL PREDICT (ALL 5 PROBLEMS)
# ═══════════════════════════════════════════════════════════════
elif page == PNAMES[8]:
    page_header("All 5 ML Problems","🔢 Manual Predict",
                "Enter values manually → AI generates predictions using trained models")

    t1,t2,t3,t4,t5 = st.tabs([
        "⚖️ P1 Yield","🌿 P2 Variety",
        "🧪 P3 Fertilizer","🔮 P4 Pre-Harvest","🚨 P5 Risk"
    ])

    # ── TAB 1 ────────────────────────────────────────────────
    with t1:
        st.markdown(f"**{L['p1_desc']}**")
        c1,c2,c3,c4 = st.columns(4)
        t1_blk  = c1.selectbox(L["block"],  BL, key="t1blk")
        t1_var  = c2.selectbox(L["variety"], VA, key="t1var")
        t1_ha   = c3.selectbox(L["hectares"],[1,2,3,4,5,6],index=2,key="t1ha")
        t1_soil = c4.selectbox(L["soil"],    ST, key="t1soil")
        c5,c6 = st.columns(2)
        t1_nurs = c5.selectbox(L["nursery"], NU, key="t1nurs")
        t1_seed = c6.number_input("Seedrate (Kg)",5,200,150,key="t1seed")
        c1,c2,c3 = st.columns(3)
        t1_dap  = c1.number_input(L["dap"],   0,400,int(t1_ha*40), key="t1dap")
        t1_urea = c2.number_input(L["urea"],  0,400,int(t1_ha*27), key="t1urea")
        t1_pot  = c3.number_input(L["potash"],0,200,int(t1_ha*10), key="t1pot")
        c4,c5,c6 = st.columns(3)
        t1_micro= c4.number_input(L["micro"], 0,200,int(t1_ha*15), key="t1micro")
        t1_weed = c5.number_input(L["weed"],  0,50, int(t1_ha*2),  key="t1weed")
        t1_pest = c6.number_input(L["pest"],  0,5000,int(t1_ha*600),key="t1pest")
        t1_bun  = st.slider(L["bundles"],80,600,int(t1_ha*85),key="t1bun")

        if st.button(L["predict"],key="t1btn",use_container_width=True):
            rf  = M_ALL["p1"]["model"]
            feats1 = M_ALL["p1"]["features"]
            le_dict = M_ALL["le_dict"]
            df_enc  = M_ALL["df_enc"]
            avg = df_enc[df_enc["Hectares"]==t1_ha].mean(numeric_only=True)
            inp = {col: avg[col] if col in avg else 0 for col in feats1}
            inp.update({
                "Hectares":t1_ha,"DAP_20days":t1_dap,"Urea_40Days":t1_urea,
                "Potassh_50Days":t1_pot,"Micronutrients_70Days":t1_micro,
                "Weed28D_thiobencarb":t1_weed,"Pest_60Day(in ml)":t1_pest,
                "Trash(in bundles)":t1_bun,"Seedrate(in Kg)":t1_seed
            })
            for col,le in le_dict.items():
                if col=="Agriblock" and col in feats1: inp[col]=le.transform([t1_blk])[0]
                elif col=="Variety" and col in feats1: inp[col]=le.transform([t1_var])[0]
                elif col=="Soil Types" and col in feats1: inp[col]=le.transform([t1_soil])[0]
                elif col=="Nursery" and col in feats1: inp[col]=le.transform([t1_nurs])[0]
            pred_y = int(rf.predict(pd.DataFrame([inp])[feats1])[0])
            result_box(f"{pred_y:,}", L["predicted_yield"],
                       f"{t1_blk} · {t1_var} · {t1_ha} Ha")
            gap = pred_y - YH[t1_ha]
            if gap>=1000: st.success(L["excellent"])
            elif gap>=-500: st.warning(L["average"])
            else: st.error(L["underperform"])

    # ── TAB 2 ────────────────────────────────────────────────
    with t2:
        st.markdown(f"**{L['p2_desc']}**")
        c1,c2,c3 = st.columns(3)
        t2_blk  = c1.selectbox(L["block"],  BL, key="t2blk")
        t2_soil = c2.selectbox(L["soil"],   ST, key="t2soil")
        t2_ha   = c3.selectbox(L["hectares"],[1,2,3,4,5,6],index=2,key="t2ha")

        if st.button(L["recommend"],key="t2btn",use_container_width=True):
            m2 = M_ALL["p2"]
            le_dict = M_ALL["le_dict"]
            df_enc  = M_ALL["df_enc"]
            feat2   = m2["features"]
            avg2    = df_enc[df_enc["Hectares"]==t2_ha].mean(numeric_only=True)
            inp2    = {col: avg2[col] if col in avg2 else 0 for col in feat2}
            inp2["Hectares"] = t2_ha
            for col,le in le_dict.items():
                if col=="Agriblock" and col in feat2: inp2[col]=le.transform([t2_blk])[0]
                elif col=="Soil Types" and col in feat2: inp2[col]=le.transform([t2_soil])[0]
            pred_v = m2["le"].inverse_transform(
                [m2["model"].predict(pd.DataFrame([inp2])[feat2])[0]])[0]
            result_box(pred_v, L["recommended_variety"],
                       f"{t2_blk} · {t2_soil} · {t2_ha} Ha · Acc {m2['acc']:.1%}")
            solution_box("✅",f"{pred_v} is best for {t2_blk}",
                "Use certified seeds. Apply DAP on Day 20 without delay.","#2d6a1f")

    # ── TAB 3 ────────────────────────────────────────────────
    with t3:
        st.markdown(f"**{L['p3_desc']}**")
        c1,c2,c3 = st.columns(3)
        t3_ha   = c1.selectbox(L["hectares"],[1,2,3,4,5,6],index=2,key="t3ha")
        t3_blk  = c2.selectbox(L["block"],   BL, key="t3blk")
        t3_soil = c3.selectbox(L["soil"],    ST, key="t3soil")

        if st.button(L["calc_dose"],key="t3btn",use_container_width=True):
            m3 = M_ALL["p3"]
            le_dict = M_ALL["le_dict"]
            df_enc  = M_ALL["df_enc"]
            feat3   = m3["features"]
            avg3    = df_enc[df_enc["Hectares"]==t3_ha].mean(numeric_only=True)
            inp3    = {col: avg3[col] if col in avg3 else 0 for col in feat3}
            inp3["Hectares"] = t3_ha
            for col,le in le_dict.items():
                if col=="Agriblock" and col in feat3: inp3[col]=le.transform([t3_blk])[0]
                elif col=="Soil Types" and col in feat3: inp3[col]=le.transform([t3_soil])[0]
            X3_in = pd.DataFrame([inp3])[feat3]
            cols_t3 = st.columns(3)
            icons3 = ["🌱","🌿","⚗️","⚗️","🍃","🐛"]
            units3 = {"DAP_20days":"Kg","Weed28D_thiobencarb":"Kg","Urea_40Days":"Kg",
                      "Potassh_50Days":"Kg","Micronutrients_70Days":"Kg","Pest_60Day(in ml)":"ml"}
            for i,(target,mdl) in enumerate(m3["models"].items()):
                rec = mdl.predict(X3_in)[0]
                u = units3.get(target,"Kg")
                cols_t3[i%3].metric(f"{icons3[i]} {target.split('(')[0]}",f"{rec} {u}")

    # ── TAB 4 ────────────────────────────────────────────────
    with t4:
        st.markdown(f"**{L['p4_desc']}**")
        c1,c2 = st.columns(2)
        t4_bun = c1.slider(L["bundles"],80,600,300,key="t4bun")
        t4_ha  = c2.selectbox(L["hectares"],[1,2,3,4,5,6],index=2,key="t4ha")

        live4 = int(M_ALL["p4"]["model"].predict(
            pd.DataFrame({"Trash(in bundles)":[t4_bun],"Hectares":[t4_ha]}))[0])
        st.markdown(f"""
<div style="background:#e8f5e9;border:1px solid #a5c8a5;border-radius:10px;
padding:10px 16px;font-size:13px;font-weight:700;color:#1a3d12;margin:8px 0;">
📦 Live: ~{live4:,} Kg</div>""", unsafe_allow_html=True)

        if st.button(L["estimate"],key="t4btn",use_container_width=True):
            result_box(f"{live4:,}", L["estimated_yield"],
                       f"{t4_bun} bundles · {t4_ha} Ha · MAE ±{M_ALL['p4']['mae']:.0f} Kg")
            exp4 = YH[t4_ha]
            if live4>=exp4: st.success(L["excellent"])
            else: st.warning(f"{L['average']} — Expected {exp4:,} Kg")

    # ── TAB 5 ────────────────────────────────────────────────
    with t5:
        st.markdown(f"**{L['p5_desc']}**")
        c1,c2,c3 = st.columns(3)
        t5_blk = c1.selectbox(L["block"],  BL, key="t5blk")
        t5_ha  = c2.selectbox(L["hectares"],[1,2,3,4,5,6],index=2,key="t5ha")
        t5_yld = c3.number_input(L["actual_yield"],1000,50000,18000,key="t5yld")
        c4,c5,c6 = st.columns(3)
        t5_dap  = c4.number_input(L["dap"],   0,400,int(t5_ha*40),key="t5dap")
        t5_urea = c5.number_input(L["urea"],  0,400,int(t5_ha*27),key="t5urea")
        t5_weed = c6.number_input(L["weed"],  0,50, int(t5_ha*2), key="t5weed")

        if st.button(L["check_risk"],key="t5btn",use_container_width=True):
            exp5  = YH[t5_ha]
            gap5  = t5_yld - exp5
            dap_ok  = t5_dap >= t5_ha*36
            urea_ok = t5_urea >= t5_ha*23
            weed_ok = t5_weed >= t5_ha*1.5
            risk_score = 0
            if gap5 < -1500: risk_score += 3
            elif gap5 < 0:   risk_score += 1
            if not dap_ok:   risk_score += 2
            if not urea_ok:  risk_score += 1
            if not weed_ok:  risk_score += 1

            if risk_score >= 4:
                st.error(f"⚠️ {L['risky']} — Risk Score: {risk_score}/7")
            elif risk_score >= 2:
                st.warning(f"🟡 Moderate Risk — Score: {risk_score}/7")
            else:
                st.success(f"✅ {L['safe']} — Score: {risk_score}/7")

            items = [
                (f"Yield gap: {gap5:+,} Kg vs expected {exp5:,} Kg",
                 "#c62828" if gap5<-1500 else "#e65100" if gap5<0 else "#2d6a1f"),
                (f"DAP: {t5_dap} Kg {'✅ OK' if dap_ok else f'❌ Need {int(t5_ha*40)} Kg on Day 20!'}",
                 "#2d6a1f" if dap_ok else "#c62828"),
                (f"Urea: {t5_urea} Kg {'✅ OK' if urea_ok else f'❌ Need {int(t5_ha*27)} Kg'}",
                 "#2d6a1f" if urea_ok else "#e65100"),
                (f"Weed: {t5_weed} Kg {'✅ OK' if weed_ok else f'❌ Need {int(t5_ha*2)} Kg'}",
                 "#2d6a1f" if weed_ok else "#e65100"),
            ]
            for txt,col in items:
                st.markdown(f"""
<div style="background:#fff;border:1px solid #d6e8d6;border-left:4px solid {col};
border-radius:10px;padding:9px 14px;margin-bottom:7px;font-size:13px;color:#2e3d2e;">
{txt}</div>""", unsafe_allow_html=True)
