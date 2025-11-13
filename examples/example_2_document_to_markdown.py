"""
مثال 2: تحويل مستند إلى Markdown
===================================

هذا المثال يوضح كيفية تحويل مستند كامل (صورة أو PDF) إلى Markdown
مع الحفاظ على التنسيق والجداول والعناوين.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
from pathlib import Path

def document_to_markdown_example():
    """
    تحويل مستند إلى Markdown مع الحفاظ على التنسيق
    """
    
    print("=" * 60)
    print("مثال 2: تحويل مستند إلى Markdown")
    print("=" * 60)
    
    # الخطوة 1: إعداد البيئة
    print("\n[1/5] إعداد البيئة...")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    # الخطوة 2: تحميل النموذج
    print("\n[2/5] تحميل النموذج...")
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
    
    model = model.eval().cuda().to(torch.bfloat16)
    print("✓ تم تحميل النموذج")
    
    # الخطوة 3: إعداد المدخلات
    print("\n[3/5] إعداد المدخلات...")
    
    # استخدام وضع Grounding للحصول على تنسيق كامل
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    # مسار المستند
    document_path = 'path/to/your/document.jpg'  # أو .png أو .pdf
    
    # مسار الإخراج
    output_path = './output/markdown_conversion'
    os.makedirs(output_path, exist_ok=True)
    
    print(f"✓ المستند: {document_path}")
    print(f"✓ الإخراج: {output_path}")
    
    # الخطوة 4: معالجة المستند
    print("\n[4/5] معالجة المستند...")
    print("هذا قد يستغرق بضع دقائق حسب حجم المستند...")
    
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=document_path,
        output_path=output_path,
        base_size=1024,       # حجم قياسي للمستندات
        image_size=640,       # حجم القطع
        crop_mode=True,       # تفعيل القص الديناميكي
        save_results=True,    # حفظ النتائج
        test_compress=True    # اختبار الضغط
    )
    
    print("✓ تمت المعالجة")
    
    # الخطوة 5: حفظ وعرض النتائج
    print("\n[5/5] حفظ النتائج...")
    
    # حفظ في ملف Markdown
    output_file = Path(output_path) / 'document.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"✓ تم حفظ الملف: {output_file}")
    
    # عرض معاينة
    print("\n" + "=" * 60)
    print("معاينة النتيجة (أول 500 حرف):")
    print("=" * 60)
    print(result[:500])
    if len(result) > 500:
        print("\n... (تم اختصار النتيجة)")
    print("=" * 60)
    
    # إحصائيات
    print("\n📊 إحصائيات:")
    print(f"   - عدد الأحرف: {len(result)}")
    print(f"   - عدد الأسطر: {result.count(chr(10))}")
    print(f"   - عدد الجداول: {result.count('|')}")
    print(f"   - عدد العناوين: {result.count('#')}")
    
    return result, output_file


def process_multiple_documents(document_paths):
    """
    معالجة عدة مستندات دفعة واحدة
    
    المعاملات:
    - document_paths: قائمة بمسارات المستندات
    """
    
    print("\n" + "=" * 60)
    print(f"معالجة {len(document_paths)} مستند")
    print("=" * 60)
    
    # تحميل النموذج مرة واحدة
    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    
    results = []
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    for i, doc_path in enumerate(document_paths, 1):
        print(f"\n[{i}/{len(document_paths)}] معالجة: {doc_path}")
        
        output_path = f'./output/batch_conversion/doc_{i}'
        os.makedirs(output_path, exist_ok=True)
        
        try:
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=doc_path,
                output_path=output_path,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True
            )
            
            # حفظ النتيجة
            output_file = Path(output_path) / f'document_{i}.md'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            
            results.append({
                'path': doc_path,
                'result': result,
                'output': output_file,
                'status': 'success'
            })
            
            print(f"✓ تم بنجاح ({len(result)} حرف)")
            
        except Exception as e:
            print(f"✗ فشل: {str(e)}")
            results.append({
                'path': doc_path,
                'result': None,
                'output': None,
                'status': 'failed',
                'error': str(e)
            })
    
    # ملخص
    print("\n" + "=" * 60)
    print("ملخص المعالجة:")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"✓ نجح: {successful}")
    print(f"✗ فشل: {failed}")
    
    return results


def main():
    """
    الدالة الرئيسية
    """
    
    print("\n🚀 بدء التشغيل...\n")
    
    # مثال 1: معالجة مستند واحد
    print("=" * 60)
    print("المثال 1: معالجة مستند واحد")
    print("=" * 60)
    
    try:
        result, output_file = document_to_markdown_example()
        print(f"\n✅ تم بنجاح! الملف: {output_file}")
    except Exception as e:
        print(f"\n❌ خطأ: {str(e)}")
    
    # مثال 2: معالجة عدة مستندات (اختياري)
    # قم بإلغاء التعليق لتجربة المعالجة الدفعية
    """
    print("\n\n" + "=" * 60)
    print("المثال 2: معالجة عدة مستندات")
    print("=" * 60)
    
    document_paths = [
        'path/to/document1.jpg',
        'path/to/document2.jpg',
        'path/to/document3.jpg',
    ]
    
    try:
        results = process_multiple_documents(document_paths)
        print("\n✅ تمت معالجة جميع المستندات!")
    except Exception as e:
        print(f"\n❌ خطأ: {str(e)}")
    """


if __name__ == "__main__":
    main()


"""
شرح تفصيلي للمخرجات:
======================

1. ملف result.mmd:
   - يحتوي على النص الكامل بصيغة Markdown
   - يتضمن العناوين والجداول والقوائم
   - يحتوي على إشارات للصور المستخرجة

2. مجلد images/:
   - يحتوي على الصور المستخرجة من المستند
   - كل صورة لها رقم تسلسلي
   - يتم الإشارة إليها في ملف Markdown

3. ملف result_with_boxes.jpg:
   - صورة المستند الأصلي مع مربعات حول العناصر
   - يوضح ما تم التعرف عليه
   - مفيد للتحقق من الدقة

مثال على المخرجات:
===================

# عنوان المستند

هذا نص عادي في المستند.

## قسم فرعي

| العمود 1 | العمود 2 | العمود 3 |
|---------|---------|---------|
| بيانات 1 | بيانات 2 | بيانات 3 |
| بيانات 4 | بيانات 5 | بيانات 6 |

![صورة مستخرجة](images/0.jpg)

### قائمة نقطية:
- النقطة الأولى
- النقطة الثانية
- النقطة الثالثة

**نص عريض** و *نص مائل*

نصائح للحصول على أفضل النتائج:
================================

1. جودة الصورة:
   - استخدم صوراً عالية الدقة (300 DPI أو أكثر)
   - تأكد من وضوح النص
   - تجنب الظلال والانعكاسات

2. إعدادات النموذج:
   - للمستندات البسيطة: base_size=1024, crop_mode=False
   - للمستندات المعقدة: base_size=1280, crop_mode=False
   - للمستندات الكبيرة: base_size=1024, crop_mode=True

3. معالجة النتائج:
   - راجع ملف result_with_boxes.jpg للتحقق من الدقة
   - قارن النتيجة مع المستند الأصلي
   - قم بتصحيح الأخطاء يدوياً إذا لزم الأمر

4. الأداء:
   - استخدم vLLM للمستندات الكبيرة
   - قم بمعالجة المستندات دفعة واحدة إذا أمكن
   - راقب استخدام ذاكرة GPU
"""
