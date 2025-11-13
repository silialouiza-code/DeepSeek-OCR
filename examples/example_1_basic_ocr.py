"""
مثال 1: استخراج نص بسيط من صورة
====================================

هذا المثال يوضح كيفية استخدام DeepSeek-OCR لاستخراج نص من صورة بسيطة
باستخدام مكتبة Transformers.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os

def basic_ocr_example():
    """
    مثال بسيط لاستخراج نص من صورة
    """
    
    # الخطوة 1: إعداد البيئة
    print("=" * 50)
    print("مثال 1: استخراج نص بسيط من صورة")
    print("=" * 50)
    
    # تحديد GPU المستخدم
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    # الخطوة 2: تحميل النموذج
    print("\n[1/4] تحميل النموذج...")
    model_name = 'deepseek-ai/DeepSeek-OCR'
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    
    # نقل النموذج إلى GPU وتحويله إلى bfloat16
    model = model.eval().cuda().to(torch.bfloat16)
    print("✓ تم تحميل النموذج بنجاح")
    
    # الخطوة 3: إعداد المدخلات
    print("\n[2/4] إعداد المدخلات...")
    
    # الأمر النصي (Prompt)
    prompt = "<image>\nFree OCR."  # استخراج نص فقط بدون تنسيق
    
    # مسار الصورة
    image_file = 'path/to/your/image.jpg'  # غير هذا إلى مسار صورتك
    
    # مسار الإخراج
    output_path = './output/basic_ocr'
    
    print(f"✓ الأمر: {prompt}")
    print(f"✓ الصورة: {image_file}")
    
    # الخطوة 4: تشغيل النموذج
    print("\n[3/4] معالجة الصورة...")
    
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=output_path,
        base_size=512,        # حجم صغير للنصوص البسيطة
        image_size=512,
        crop_mode=False,      # بدون قص
        save_results=True,    # حفظ النتائج
        test_compress=False   # بدون اختبار الضغط
    )
    
    print("✓ تمت المعالجة بنجاح")
    
    # الخطوة 5: عرض النتائج
    print("\n[4/4] النتائج:")
    print("-" * 50)
    print(result)
    print("-" * 50)
    
    print(f"\n✓ تم حفظ النتائج في: {output_path}")
    
    return result


def main():
    """
    الدالة الرئيسية
    """
    try:
        result = basic_ocr_example()
        print("\n✅ تم تنفيذ المثال بنجاح!")
        
    except Exception as e:
        print(f"\n❌ حدث خطأ: {str(e)}")
        print("\nتأكد من:")
        print("1. تثبيت جميع المكتبات المطلوبة")
        print("2. توفر GPU مع CUDA")
        print("3. صحة مسار الصورة")


if __name__ == "__main__":
    main()


"""
ملاحظات مهمة:
==============

1. متطلبات التشغيل:
   - GPU مع CUDA 11.8 أو أحدث
   - ذاكرة GPU: 8GB على الأقل
   - Python 3.12 أو أحدث

2. المعاملات المهمة:
   - base_size: حجم الصورة الأساسي (512 للصور الصغيرة)
   - image_size: حجم القطع (512 للصور الصغيرة)
   - crop_mode: False للصور البسيطة، True للصور الكبيرة

3. الأوامر البديلة:
   - "<image>\nFree OCR." - نص فقط
   - "<image>\n<|grounding|>Convert the document to markdown." - مع تنسيق
   - "<image>\nDescribe this image in detail." - وصف الصورة

4. تحسين الأداء:
   - استخدم base_size=512 للصور الصغيرة
   - استخدم base_size=1024 للصور المتوسطة
   - استخدم base_size=1280 للصور الكبيرة

5. حل المشاكل:
   - إذا نفدت ذاكرة GPU، قلل base_size
   - إذا كانت النتائج غير دقيقة، زد base_size
   - إذا كانت المعالجة بطيئة، استخدم vLLM بدلاً من Transformers
"""
