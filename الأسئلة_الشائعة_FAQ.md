# ❓ الأسئلة الشائعة (FAQ) - DeepSeek-OCR

هذا الملف يجيب على الأسئلة الأكثر شيوعاً حول DeepSeek-OCR.

---

## 📚 أسئلة عامة

### 1. ما هو DeepSeek-OCR؟

**الجواب:**
DeepSeek-OCR هو نموذج ذكاء اصطناعي متقدم لاستخراج النصوص من الصور والمستندات (OCR) مع القدرة على:
- استخراج النصوص بدقة عالية
- تحويل المستندات إلى Markdown
- التعرف على الجداول والمعادلات الرياضية
- معالجة ملفات PDF متعددة الصفحات
- ضغط بصري للسياق (تقليل عدد الرموز البصرية)

---

### 2. هل DeepSeek-OCR مجاني؟

**الجواب:**
✅ **نعم!** DeepSeek-OCR مفتوح المصدر ومجاني تماماً.

**لكن تحتاج:**
- GPU قوية (يفضل A100 أو أفضل)
- أو استخدام خدمات سحابية (Google Colab, AWS, Azure)
- تكلفة استخدام GPU السحابية: ~$1-3/ساعة

---

### 3. هل يدعم اللغة العربية؟

**الجواب:**
✅ **نعم!** DeepSeek-OCR يدعم اللغة العربية بشكل ممتاز.

**مثال:**
```python
prompt = "<image>\nFree OCR."
result = model.infer(tokenizer, prompt=prompt, image_file='arabic_text.jpg')
# سيستخرج النص العربي بدقة عالية
```

---

### 4. ما الفرق بين DeepSeek-OCR والـ OCR التقليدي؟

**الجواب:**

| الميزة | OCR التقليدي | DeepSeek-OCR |
|--------|--------------|--------------|
| استخراج النص | ✅ | ✅ |
| فهم التخطيط | ⚠️ محدود | ✅ ممتاز |
| الجداول | ⚠️ ضعيف | ✅ ممتاز |
| المعادلات الرياضية | ❌ | ✅ |
| التحويل إلى Markdown | ❌ | ✅ |
| الذكاء الاصطناعي | ❌ | ✅ |

---

## 🔧 أسئلة التثبيت والإعداد

### 5. ما هي المتطلبات الأساسية؟

**الجواب:**

**الأجهزة:**
- GPU: NVIDIA مع CUDA 11.8 أو أحدث
- ذاكرة GPU: 8GB كحد أدنى (16GB موصى به)
- RAM: 16GB كحد أدنى
- مساحة تخزين: 20GB

**البرمجيات:**
- Python 3.12 أو أحدث
- CUDA 11.8 أو أحدث
- PyTorch 2.6.0 أو أحدث

---

### 6. كيف أثبت DeepSeek-OCR؟

**الجواب:**

```bash
# 1. إنشاء بيئة افتراضية
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# 2. تثبيت PyTorch
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# 3. تثبيت المكتبات
pip install -r requirements.txt

# 4. تثبيت Flash Attention (اختياري)
pip install flash-attn==2.7.3 --no-build-isolation
```

**راجع:** `دليل_البدء_السريع.md` للتفاصيل الكاملة.

---

### 7. هل يمكن استخدامه بدون GPU؟

**الجواب:**
⚠️ **نظرياً نعم، لكن عملياً لا.**

- النموذج كبير جداً (~10GB)
- المعالجة على CPU ستكون **بطيئة جداً** (ساعات لصورة واحدة)
- **الحل:** استخدم Google Colab مع GPU مجاني

---

### 8. كيف أستخدم Google Colab؟

**الجواب:**

```python
# في Google Colab:

# 1. تفعيل GPU
# Runtime > Change runtime type > GPU > T4 GPU

# 2. التثبيت
!pip install transformers torch
!pip install -r requirements.txt

# 3. الاستخدام
from transformers import AutoModel, AutoTokenizer
# ... باقي الكود
```

---

## 🎯 أسئلة الاستخدام

### 9. كيف أستخرج نصاً من صورة؟

**الجواب:**

```python
from transformers import AutoModel, AutoTokenizer
import torch

# تحميل النموذج
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)

# استخراج النص
prompt = "<image>\nFree OCR."
result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file='your_image.jpg',
    output_path='./output',
    base_size=512,
    image_size=512,
    crop_mode=False
)

print(result)
```

---

### 10. كيف أحول مستنداً إلى Markdown؟

**الجواب:**

```python
# استخدم وضع Grounding
prompt = "<image>\n<|grounding|>Convert the document to markdown."

result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file='document.jpg',
    output_path='./output',
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True
)

# النتيجة ستكون Markdown كامل مع الجداول والتنسيق
```

---

### 11. كيف أعالج ملف PDF؟

**الجواب:**

**الطريقة 1: باستخدام Transformers (بسيطة)**
```python
import fitz  # PyMuPDF
from PIL import Image

# تحويل PDF إلى صور
pdf = fitz.open('document.pdf')
for page_num in range(pdf.page_count):
    page = pdf[page_num]
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # معالجة كل صفحة
    result = model.infer(tokenizer, image_file=img, ...)
```

**الطريقة 2: باستخدام vLLM (أسرع)**
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
# عدّل config.py
python run_dpsk_ocr_pdf.py
```

**راجع:** `examples/example_3_pdf_processing.py`

---

### 12. ما هي الأوامر (Prompts) المتاحة؟

**الجواب:**

```python
# 1. نص فقط (بدون تنسيق)
prompt = "<image>\nFree OCR."

# 2. تحويل إلى Markdown (مع تنسيق)
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# 3. OCR عام
prompt = "<image>\n<|grounding|>OCR this image."

# 4. وصف الصورة
prompt = "<image>\nDescribe this image in detail."

# 5. تحليل الأشكال
prompt = "<image>\nParse the figure."

# 6. تحديد موقع نص
prompt = "<image>\nLocate <|ref|>النص المطلوب<|/ref|> in the image."
```

---

## ⚙️ أسئلة الإعدادات والمعاملات

### 13. ما معنى base_size و image_size؟

**الجواب:**

**base_size:** حجم الصورة الأساسي
- 512: للصور الصغيرة (64 رمز بصري)
- 640: للصور المتوسطة (100 رمز بصري)
- 1024: للصور القياسية (256 رمز بصري) - **موصى به**
- 1280: للصور الكبيرة (400 رمز بصري)

**image_size:** حجم القطع عند استخدام crop_mode=True
- عادة 640 أو 1024

**مثال:**
```python
# للصور البسيطة
base_size=512, image_size=512, crop_mode=False

# للمستندات المعقدة (موصى به)
base_size=1024, image_size=640, crop_mode=True

# للصور عالية الدقة
base_size=1280, image_size=1280, crop_mode=False
```

---

### 14. متى أستخدم crop_mode=True؟

**الجواب:**

**استخدم crop_mode=True عندما:**
- الصورة كبيرة جداً (>1024×1024)
- المستند معقد مع تفاصيل كثيرة
- تريد أفضل دقة ممكنة
- لديك ذاكرة GPU كافية

**استخدم crop_mode=False عندما:**
- الصورة صغيرة (<1024×1024)
- النص بسيط
- تريد سرعة أعلى
- ذاكرة GPU محدودة

---

### 15. كيف أختار الإعدادات المناسبة؟

**الجواب:**

**حسب نوع الصورة:**

| نوع الصورة | base_size | image_size | crop_mode |
|------------|-----------|------------|-----------|
| نص بسيط | 512 | 512 | False |
| مستند عادي | 1024 | 640 | True |
| مستند معقد | 1280 | 1280 | False |
| فاتورة | 1024 | 640 | True |
| كتاب | 1024 | 1024 | False |
| جدول | 1024 | 1024 | False |

---

## 🐛 أسئلة حل المشاكل

### 16. خطأ: "CUDA out of memory"

**الجواب:**

**الحلول:**

```python
# 1. قلل حجم الصورة
base_size = 512  # بدلاً من 1024

# 2. عطّل crop_mode
crop_mode = False

# 3. قلل استخدام الذاكرة (vLLM)
gpu_memory_utilization = 0.6  # بدلاً من 0.9

# 4. قلل MAX_CONCURRENCY (vLLM)
MAX_CONCURRENCY = 50  # بدلاً من 100

# 5. استخدم GPU أكبر
# أو عالج الصور على دفعات أصغر
```

---

### 17. النتائج غير دقيقة

**الجواب:**

**الحلول:**

```python
# 1. زد حجم الصورة
base_size = 1280  # بدلاً من 512

# 2. استخدم وضع Grounding
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# 3. حسّن جودة الصورة المدخلة
# - استخدم DPI أعلى (300+)
# - تأكد من وضوح النص
# - تجنب الظلال والانعكاسات

# 4. فعّل crop_mode للصور الكبيرة
crop_mode = True
```

---

### 18. المعالجة بطيئة جداً

**الجواب:**

**الحلول:**

```python
# 1. استخدم vLLM بدلاً من Transformers
# vLLM أسرع 5-10 مرات

# 2. قلل حجم الصورة
base_size = 512  # للصور البسيطة

# 3. استخدم المعالجة الدفعية (vLLM)
batch_inputs = [prepare_image(img) for img in images]
outputs = llm.generate(batch_inputs)

# 4. عطّل crop_mode
crop_mode = False
```

---

### 19. خطأ: "Model not found"

**الجواب:**

**الحلول:**

```python
# 1. تأكد من الاتصال بالإنترنت
# النموذج سيُحمّل تلقائياً من Hugging Face

# 2. حمّل النموذج يدوياً
from huggingface_hub import snapshot_download
snapshot_download(repo_id="deepseek-ai/DeepSeek-OCR")

# 3. استخدم مسار محلي
model_name = '/path/to/local/model'
```

---

### 20. خطأ: "Import error"

**الجواب:**

```bash
# 1. أعد تثبيت المكتبات
pip install --upgrade transformers torch

# 2. تأكد من تثبيت جميع المتطلبات
pip install -r requirements.txt

# 3. تحقق من إصدار Python
python --version  # يجب أن يكون 3.12+

# 4. أعد إنشاء البيئة الافتراضية
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

---

## 🚀 أسئلة الأداء

### 21. كم تستغرق معالجة صورة واحدة؟

**الجواب:**

**باستخدام Transformers:**
- صورة صغيرة (512×512): 5-10 ثواني
- صورة متوسطة (1024×1024): 15-30 ثانية
- صورة كبيرة (1280×1280): 30-60 ثانية

**باستخدام vLLM:**
- صورة صغيرة: 2-5 ثواني
- صورة متوسطة: 5-10 ثواني
- صورة كبيرة: 10-20 ثانية

**ملاحظة:** الأوقات تعتمد على GPU المستخدم.

---

### 22. كم صورة يمكن معالجتها في الساعة؟

**الجواب:**

**باستخدام Transformers (A100):**
- ~100-200 صورة/ساعة

**باستخدام vLLM (A100):**
- ~500-1000 صورة/ساعة

**مع المعالجة الدفعية (vLLM):**
- ~1000-2000 صورة/ساعة

---

### 23. كيف أحسّن الأداء؟

**الجواب:**

**للسرعة:**
```python
# 1. استخدم vLLM
# 2. قلل base_size
# 3. عطّل crop_mode
# 4. استخدم المعالجة الدفعية
# 5. استخدم GPU أقوى
```

**للدقة:**
```python
# 1. زد base_size
# 2. فعّل crop_mode
# 3. استخدم وضع Grounding
# 4. حسّن جودة الصورة المدخلة
```

**للتوازن:**
```python
base_size = 1024
image_size = 640
crop_mode = True
# استخدم vLLM
```

---

## 💰 أسئلة التكلفة

### 24. كم تكلفة استخدام DeepSeek-OCR؟

**الجواب:**

**النموذج نفسه: مجاني! ✅**

**لكن تحتاج GPU:**

| الخيار | التكلفة | المناسب لـ |
|--------|---------|-----------|
| GPU محلي | $0 (إذا كان لديك) | الاستخدام المكثف |
| Google Colab (مجاني) | $0 | التجربة والتعلم |
| Google Colab Pro | $10/شهر | الاستخدام المتوسط |
| AWS/Azure (A100) | $1-3/ساعة | الإنتاج |

**مثال:**
- معالجة 1000 صورة بـ vLLM على A100
- الوقت: ~1 ساعة
- التكلفة: ~$2-3

---

### 25. هل يمكن استخدامه مجاناً؟

**الجواب:**
✅ **نعم!**

**الخيارات المجانية:**
1. **Google Colab** (مجاني):
   - GPU T4 مجاني
   - محدود بـ 12 ساعة/يوم
   - مناسب للتجربة والتعلم

2. **Kaggle** (مجاني):
   - GPU P100 مجاني
   - 30 ساعة/أسبوع
   - مناسب للمشاريع الصغيرة

3. **GPU محلي** (إذا كان لديك):
   - مجاني تماماً
   - بدون قيود

---

## 📊 أسئلة المقارنة

### 26. ما الفرق بين Transformers و vLLM؟

**الجواب:**

| الميزة | Transformers | vLLM |
|--------|-------------|------|
| السرعة | بطيء | سريع جداً (5-10x) |
| السهولة | سهل جداً | متوسط |
| المعالجة الدفعية | محدودة | ممتازة |
| التثبيت | بسيط | معقد نسبياً |

**راجع:** `مقارنة_الطرق_والأداء.md` للتفاصيل الكاملة.

---

### 27. أيهما أفضل للمبتدئين؟

**الجواب:**
**Transformers** بدون شك!

**الأسباب:**
- ✅ سهل التثبيت
- ✅ كود بسيط وواضح
- ✅ توثيق ممتاز
- ✅ مناسب للتعلم

**ابدأ بـ Transformers، ثم انتقل إلى vLLM عندما تحتاج سرعة أعلى.**

---

## 🎓 أسئلة التعلم

### 28. من أين أبدأ؟

**الجواب:**

**المسار الموصى به:**

```
الأسبوع 1:
├─ اقرأ: دليل_البدء_السريع.md
├─ ثبّت المكتبات
└─ جرب: examples/example_1_basic_ocr.py

الأسبوع 2:
├─ اقرأ: شرح_DeepSeek_OCR_كامل.md
├─ جرب: examples/example_2_document_to_markdown.py
└─ جرب صوراً مختلفة

الأسبوع 3:
├─ اقرأ: مقارنة_الطرق_والأداء.md
├─ جرب: examples/example_3_pdf_processing.py
└─ انتقل إلى vLLM

الأسبوع 4+:
├─ طوّر مشاريعك الخاصة
├─ حسّن الأداء
└─ شارك مع المجتمع
```

---

### 29. ما هي أفضل المصادر للتعلم؟

**الجواب:**

**في هذا المشروع:**
1. `دليل_البدء_السريع.md` - للبدء السريع
2. `شرح_DeepSeek_OCR_كامل.md` - شرح شامل
3. `examples/` - أمثلة عملية
4. `مقارنة_الطرق_والأداء.md` - مقارنات
5. `الأسئلة_الشائعة_FAQ.md` - هذا الملف

**مصادر خارجية:**
- الورقة البحثية: https://arxiv.org/abs/2510.18234
- Hugging Face: https://huggingface.co/deepseek-ai/DeepSeek-OCR
- GitHub: https://github.com/deepseek-ai/DeepSeek-OCR
- Discord: https://discord.gg/Tc7c45Zzu5

---

### 30. كيف أطور مهاراتي؟

**الجواب:**

**خطوات عملية:**

1. **ابدأ بالأمثلة البسيطة**
   - جرب example_1
   - افهم الكود
   - عدّل المعاملات

2. **جرب سيناريوهات مختلفة**
   - صور مختلفة
   - مستندات معقدة
   - ملفات PDF

3. **تعلم التحسين**
   - جرب إعدادات مختلفة
   - قارن الأداء
   - راقب استخدام الموارد

4. **طور مشاريع**
   - تطبيق OCR خاص
   - أداة تحويل PDF
   - نظام رقمنة مستندات

5. **شارك وتعلم**
   - شارك مشاريعك
   - ساعد الآخرين
   - تعلم من المجتمع

---

## 🤝 أسئلة المجتمع

### 31. كيف أحصل على مساعدة؟

**الجواب:**

**قنوات الدعم:**

1. **Discord الرسمي:**
   - https://discord.gg/Tc7c45Zzu5
   - مجتمع نشط
   - دعم سريع

2. **GitHub Issues:**
   - https://github.com/deepseek-ai/DeepSeek-OCR/issues
   - للمشاكل التقنية
   - طلبات الميزات

3. **Hugging Face:**
   - https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions
   - نقاشات عامة
   - أسئلة وأجوبة

---

### 32. كيف أساهم في المشروع؟

**الجواب:**

**طرق المساهمة:**

1. **الإبلاغ عن المشاكل:**
   - افتح Issue على GitHub
   - وصف المشكلة بالتفصيل
   - أرفق أمثلة

2. **تحسين التوثيق:**
   - أضف أمثلة جديدة
   - حسّن الشروحات
   - ترجم إلى لغات أخرى

3. **تطوير الكود:**
   - أصلح الأخطاء
   - أضف ميزات جديدة
   - حسّن الأداء

4. **مشاركة التجارب:**
   - اكتب مقالات
   - أنشئ فيديوهات تعليمية
   - شارك مشاريعك

---

## 📞 هل لديك سؤال آخر؟

إذا لم تجد إجابة لسؤالك هنا:

1. **ابحث في الملفات الأخرى:**
   - `شرح_DeepSeek_OCR_كامل.md`
   - `دليل_البدء_السريع.md`
   - `مقارنة_الطرق_والأداء.md`

2. **اسأل المجتمع:**
   - Discord: https://discord.gg/Tc7c45Zzu5
   - GitHub: https://github.com/deepseek-ai/DeepSeek-OCR

3. **راجع التوثيق الرسمي:**
   - README.md
   - الورقة البحثية

---

**نتمنى لك تجربة ممتعة مع DeepSeek-OCR! 🎉**
