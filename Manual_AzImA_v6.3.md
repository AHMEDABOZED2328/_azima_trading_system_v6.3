# دليل المستخدم - نظام AzImA Trading System v6.3

## نظرة عامة

هذا الدليل يوضح ملفات المشروع الأساسية، وظيفتها، وطريقة تشغيل النظام من البداية للنهاية. النظام مصمم لتدريب نماذج ذكاء اصطناعي على بيانات الأسواق المالية (EURUSD) واختبارها.

## شجرة الملفات (File Structure)

```text
azima_trading_system_v6.3/
├── config_v6.py                # ملف الإعدادات الرئيسي (التحكم في كل المتغيرات)
├── logger.py                   # نظام تسجيل الأحداث (Logs)
├── 0--convert_new_data.py      # تحويل البيانات الخام وتجهيزها
├── feature_engineering_v6.py   # هندسة الميزات (إضافة المؤشرات الفنية)
├── triple_barrier_labeling.py  # تصنيف البيانات (Triple Barrier Method)
├── prepare_data_complete.py    # تجميع خطوات التجهيز في سكربت واحد
│
├── train_base_models_v6.py     # تدريب النماذج الأساسية (LSTM, XGBoost, etc.)
├── train_ensemble_v6.py        # تدريب نموذج التجميع (Ensemble)
├── train_filter_model.py       # تدريب نموذج الفلتر (لتقليل الصفقات الخاسرة)
├── run_full_system_v6.py       # تشغيل النظام بالكامل (تدريب + اختبار)
│
├── backtest_v6.py              # نظام الباك تيست الرئيسي
├── paper_trading_v6.py         # نظام التداول الورقي (المحاكاة الحية)
│
├── -- gpt_backtest/            
│   └── backtest_macd_rsi_basket.py  # باك تيست استراتيجية السلة (معدل لاستخدام إشارات التدريب)
│
├── models_v6/                  # المجلد الذي تحفظ فيه النماذج المدربة
├── results_v6/                 # نتائج الاختبارات والتقارير
└── logs/                       # ملفات السجل (Logs)
```

---

## تفاصيل الملفات الأساسية

| اسم الملف | الوظيفة التفصيلية |
| :--- | :--- |
| **config_v6.py** | **القلب النابض للنظام.** يحتوي على كل إعدادات النماذج، المسارات، الفترات الزمنية، وإعدادات التداول. يجب مراجعته قبل أي تشغيل. |
| **prepare_data_complete.py** | يقوم بقراءة البيانات الخام، حساب المؤشرات الفنية، وتطبيق التصنيف (Labeling). ينتج ملف `data/processed/ready_for_training.csv`. |
| **train_base_models_v6.py** | يدرب مجموعة من النماذج المختلفة (مثل LSTM, Random Forest, XGBoost) ويحفظها في مجلد `models_v6`. |
| **train_ensemble_v6.py** | يأخذ توقعات النماذج الأساسية ويدمجها في نموذج واحد أقوى (Ensemble Model). |
| **backtest_v6.py** | يقوم باختبار الاستراتيجية على بيانات تاريخية ويصدر تقرير بالأداء (Equity Curve, Trades List). |
| **gpt_backtest/backtest_macd_rsi_basket.py** | سكربت باك تيست خاص باستراتيجية السلة. تم تعديله ليقرأ إشارات الدخول من ملف التدريب (Target) مباشرة. |

---

## دليل التشغيل (خطوة بخطوة)

### 1. تجهيز البيانات

تأكد من وجود البيانات الخام (مثل `007_(Data_Cleaning_Code).csv`) ثم شغل:

```bash
python prepare_data_complete.py
```

*المخرج:* ملف `data/processed/final_labeled_data.csv` جاهز للتدريب.

### 2. تدريب النماذج

لتدريب النظام بالكامل دفعة واحدة (النماذج الأساسية + المجمع + الفلتر):

```bash
python run_full_system_v6.py
```

*أو يمكنك تشغيل كل خطوة على حدة:*

```bash
python train_base_models_v6.py
python train_ensemble_v6.py
```

### 3. تشغيل الباك تيست (Backtest)

لاختبار الأداء العام للنظام المجمع:

```bash
python backtest_v6.py
```

*النتائج:* ستجدها في مجلد `results_v6`.

### 4. تشغيل باك تيست استراتيجية السلة (السكربت المعدل)

هذا السكربت يستخدم إشارات التدريب (Target) للدخول في صفقات سلة (Basket):

```bash
cd "-- gpt_backtest"
python backtest_macd_rsi_basket.py
```

*ملاحظة:* يمكنك تعديل تاريخ البداية والنهاية داخل هذا الملف عبر متغيرات `START_DATE` و `END_DATE`.

---

## المخرجات (Outputs)

- **models_v6/**: يحتوي على ملفات `.pkl` و `.h5` للنماذج المدربة.
- **results_v6/**:
  - `trades.csv`: سجل بكل الصفقات (توقيت الدخول، الخروج، الربح/الخسارة).
  - `equity_curve.csv`: تطور رصيد الحساب مع الوقت.
  - `metrics.json`: مقاييس الأداء (Win Rate, Sharpes Ratio, Drawdown).
- **gpt_backtest/**: (عند تشغيل السكربت الخاص به)
  - `data_with_signals.csv`: البيانات مع الإشارات المنفذة.
  - `equity_curve.csv` و `trades.csv` الخاصة بهذا الاختبار فقط.
