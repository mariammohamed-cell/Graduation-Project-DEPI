import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle

# --- 1. تحميل النماذج والمحولات ---
@st.cache_resource
def load_assets():
    try:
        # تحميل نموذج LightGBM
        model = lgb.Booster(model_file='lgbm_model.txt') # تأكد من أن المسار والاسم صحيحان
        # st.cache_resource does not work well with model loading in lgb
        
        # تحميل قائمة الميزات المطلوبة
        with open('selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)
            
        # تحميل محول التسميات (Label Encoder)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
            
        return model, selected_features, le
    except FileNotFoundError as e:
        st.error(f"خطأ في تحميل الملفات: تأكد من وجود ملفات 'lgbm_model.txt'، 'selected_features.pkl'، و 'label_encoder.pkl' في نفس مجلد التطبيق.")
        st.stop()
    except Exception as e:
        st.error(f"حدث خطأ غير متوقع أثناء تحميل الأصول: {e}")
        st.stop()

# يتم تحميل الأصول مرة واحدة
model, selected_features, le = load_assets()

# --- 2. إعداد واجهة المستخدم (UI) ---
st.set_page_config(page_title="تنبؤ بخطورة حوادث الطرق", layout="wide")
st.title("🚦 نظام التنبؤ بخطورة حوادث الطرق")
st.markdown("يرجى إدخال البيانات المطلوبة للتنبؤ بخطورة الحادث (خفيف، متوسط، خطير).")

# بناء الأعمدة لتنظيم المدخلات
col1, col2, col3 = st.columns(3)

# قاموس لتخزين المدخلات
user_inputs = {}

# العمود الأول: خصائص الطريق والسرعة
with col1:
    st.header("بيانات الطريق")
    user_inputs['Speed_limit'] = st.slider("حد السرعة (ميل/ساعة)", 20, 70, 40)
    user_inputs['Urban_or_Rural_Area'] = st.selectbox("المنطقة", options=[(1, "منطقة حضرية"), (2, "منطقة ريفية")], format_func=lambda x: x[1])[0]
    user_inputs['Road_Type'] = st.selectbox("نوع الطريق", options=[(3, "طريق مفرد (Single)"), (6, "طريق مزدوج (Dual)"), (9, "طريق دائري (Roundabout)"), (7, "مخرج (Slip Road)"), (12, "طريق باتجاه واحد (One way)")], format_func=lambda x: x[1])[0]
    
# العمود الثاني: الظروف المحيطة
with col2:
    st.header("الظروف المحيطة")
    user_inputs['Light_Conditions'] = st.selectbox("ظروف الإضاءة", options=[(1, "ضوء النهار"), (4, "ظلام - إضاءة الشارع متوفرة"), (5, "ظلام - إضاءة الشارع غير متوفرة"), (6, "ظلام - إضاءة الشارع معطوبة"), (7, "ظلام - لا توجد إضاءة للشارع")], format_func=lambda x: x[1])[0]
    user_inputs['Road_Surface_Conditions'] = st.selectbox("حالة سطح الطريق", options=[(1, "جاف"), (2, "رطب/مبلل"), (3, "ثلج/جليد"), (4, "طين/أتربة")], format_func=lambda x: x[1])[0]
    user_inputs['Day_of_Week'] = st.selectbox("يوم الأسبوع", options=[(1, "الأحد"), (2, "الاثنين"), (3, "الثلاثاء"), (4, "الأربعاء"), (5, "الخميس"), (6, "الجمعة"), (7, "السبت")], format_func=lambda x: x[1])[0]

# العمود الثالث: تفاصيل الحادث والوقت
with col3:
    st.header("تفاصيل أخرى")
    user_inputs['Did_Police_Officer_Attend_Scene_of_Accident'] = st.selectbox("حضور الشرطة", options=[(1, "نعم"), (2, "لا")], format_func=lambda x: x[1])[0]
    user_inputs['2nd_Road_Class'] = st.selectbox("تصنيف الطريق الثانوي", options=[(1, "A"), (2, "B"), (3, "C"), (4, "الطرق المحلية")], format_func=lambda x: x[1])[0]
    time_input = st.time_input("وقت وقوع الحادث")
    # استخراج الساعة فقط
    user_inputs['Accident_Hour'] = time_input.hour

# --- 3. بناء المدخلات وهندسة الميزات ---

# تحويل المدخلات إلى تنسيق قائمة (List of one element) لتناسب DataFrame
all_features_data = {k: [v] for k, v in user_inputs.items()}

# هندسة الميزات (Feature Engineering)
# 1. التفاعل بين السرعة والمنطقة
all_features_data['Speed_Urban_Rural'] = [all_features_data['Urban_or_Rural_Area'][0] * all_features_data['Speed_limit'][0]]
# 2. التفاعل بين الإضاءة وحالة الطريق
all_features_data['Light_Road_Interaction'] = [all_features_data['Light_Conditions'][0] * all_features_data['Road_Surface_Conditions'][0]]


# --- 4. التحضير الدقيق للبيانات لتجنب خطأ LightGBMError (الخطوة الحاسمة) ---
# 1. إنشاء DataFrame من القاموس
input_df = pd.DataFrame(all_features_data, index=[0])

# 2. ضمان وجود جميع الميزات المتوقعة (تعيين 0 للميزات المفقودة)
# هذه الخطوة مهمة جداً إذا كان النموذج يتوقع ميزات تم إنشاؤها عبر One-Hot Encoding
for col in selected_features:
    if col not in input_df.columns:
        input_df[col] = 0

# 3. خطوة حاسمة: إعادة ترتيب واختيار الميزات بناءً على قائمة selected_features
input_df_final = input_df[selected_features]

# 4. تحويل جميع الأعمدة إلى float
input_df_final = input_df_final.astype(float)


# --- 5. خطوة التنبؤ ونتائج التصحيح ---

# شريط جانبي لعرض نتائج التصحيح
st.sidebar.title("📊 معلومات التصحيح")
st.sidebar.markdown("---")
st.sidebar.caption(f"عدد الميزات المتوقع: **{len(selected_features)}**")
st.sidebar.caption(f"عدد الميزات التي تم إنشاؤها: **{input_df_final.shape[1]}**")
st.sidebar.markdown("---")


if st.button("تنبؤ بالخطورة"):
    try:
        # التحويل إلى مصفوفة NumPy
        input_np = input_df_final.to_numpy()
        
        # التحقق النهائي من الشكل
        if input_np.shape[1] != len(selected_features):
             raise ValueError(f"عدم تطابق في الشكل: المتوقع {len(selected_features)} ميزة، تم الحصول على {input_np.shape[1]}")
        
        # التنبؤ بالاحتمالات (الخطوة التي كانت تفشل)
        probs = model.predict(input_np)
        
        # تحديد الفئة ذات الاحتمالية الأعلى
        pred = np.argmax(probs, axis=1)
        
        # تحويل التنبؤ الرقمي إلى التسمية الأصلية
        pred_label_raw = le.inverse_transform(pred)[0]
        
        # ربط التسميات بالمعنى
        label_map = {0: "خفيف (Slight)", 1: "خطير (Serious)", 2: "مميت (Fatal)"} # تأكد من أن التعيين يطابق تدريب نموذجك
        severity_label = label_map.get(pred[0], "غير معروف")
        
        # --- عرض النتيجة ---
        st.subheader("✅ نتيجة التنبؤ:")
        
        # تحديد الألوان حسب الخطورة
        color_map = {"خفيف (Slight)": "green", "متوسط (Serious)": "orange", "خطير (Fatal)": "red"}
        
        st.markdown(f"**من المرجح أن تكون خطورة هذا الحادث هي: ** <span style='font-size: 24px; color:{color_map.get(severity_label, 'black')}'>**{severity_label}**</span>", unsafe_allow_html=True)
        
        st.write("---")
        st.markdown("**تفاصيل الاحتمالات:**")
        # عرض احتمالات كل فئة
        
        # يتم تخزين الاحتمالات في LightGBM عادةً بالترتيب التصاعدي لـ Label Encoder
        probs_df = pd.DataFrame({
            "الخطورة": [label_map.get(i, f"الفئة {i}") for i in range(len(probs[0]))],
            "الاحتمال": probs[0]
        }).sort_values(by="الاحتمال", ascending=False)
        
        st.dataframe(probs_df.style.format({'الاحتمال': "{:.2%}"}), hide_index=True)

    except Exception as e:
        st.error(f"❌ فشل التنبؤ بسبب خطأ داخلي. يرجى مراجعة نافذة سجل الأخطاء. تفاصيل الخطأ: {e}")
