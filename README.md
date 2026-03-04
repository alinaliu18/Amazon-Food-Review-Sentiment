# Amazon Food Review Model → Book Review Testing 📚

## 项目概述

这个项目帮你测试你的**Amazon Food Review sentiment model**在**书评**上的表现，验证模型在跨领域迁移中的能力。

---

## 🎯 核心问题

**你的模型能识别书评吗？**

你的模型训练数据：Amazon **食品**评论（"delicious", "tasty", "flavor"）  
测试数据：**书籍**评论（"engaging", "plot", "character development"）

**挑战**：
- ❌ 词汇完全不同（食品 vs 书籍）
- ❌ TF-IDF权重基于食品词汇
- ❌ 跨领域迁移通常效果下降

**解决方案**：测试 + 分析 + 改进

---

## 📁 文件说明

### 1. `BookReviewTesting.ipynb` ⭐ **主文件**
完整的测试notebook，包含：
- ✅ 样本书评数据（20条真实风格的评论）
- ✅ 与你原项目**完全相同**的预处理pipeline
- ✅ 模型加载和预测
- ✅ 性能评估（accuracy, confusion matrix, classification report）
- ✅ 跨域分析（词汇对比、误分类分析）
- ✅ 改进建议
- ✅ 真实网页爬虫代码（Goodreads/Amazon Books）

**使用方法**：
```bash
jupyter notebook BookReviewTesting.ipynb
```

### 2. `book_review_scraper.py`
Python脚本版本，包含：
- `BookReviewScraper` - Goodreads和Amazon Books爬虫类
- `TextPreprocessor` - 预处理pipeline（匹配你的原项目）
- `SentimentModelTester` - 模型测试类
- `create_sample_book_reviews()` - 创建样本数据

**使用方法**：
```python
from book_review_scraper import *

# 创建样本数据
reviews = create_sample_book_reviews()

# 预处理
preprocessor = TextPreprocessor()
reviews = preprocessor.preprocess_dataframe(reviews)

# 测试模型（需要先保存vectorizer！）
tester = SentimentModelTester(
    model_path='amazon_sentiment_final_lr_tfidf.pkl',
    vectorizer_path='tfidf_vectorizer.pkl'
)
results = tester.predict_sentiment(reviews)
tester.evaluate_performance(results)
```

### 3. `save_vectorizer_fix.py` ⚠️ **重要！**
你的原项目**缺少**保存TF-IDF vectorizer的步骤！

**问题**：你只保存了模型，没保存vectorizer
**后果**：无法在新数据上测试
**解决**：运行这个脚本，它会告诉你如何修复

```bash
python save_vectorizer_fix.py
```

---

## 🚀 完整使用流程

### Step 1: 修复原项目（保存vectorizer）

**回到你的 `AmazonFoodReviewProject.ipynb`：**

找到创建TF-IDF vectorizer的cell：
```python
# 你原来的代码
vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train['Cleaned_Review'])

# 🔴 添加这一行！
import joblib
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
```

运行cell，确保生成了 `tfidf_vectorizer.pkl` 文件。

### Step 2: 测试准备

确保你有这些文件：
```
✅ amazon_sentiment_final_lr_tfidf.pkl  (模型)
✅ tfidf_vectorizer.pkl                 (vectorizer - 刚保存的)
✅ BookReviewTesting.ipynb              (测试notebook)
```

### Step 3: 运行测试

**Option A - Jupyter Notebook（推荐）**：
```bash
jupyter notebook BookReviewTesting.ipynb
```
按顺序运行所有cells。

**Option B - Python脚本**：
```python
from book_review_scraper import *
import joblib

# 加载模型和vectorizer
model = joblib.load('amazon_sentiment_final_lr_tfidf.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# 创建测试数据
reviews = create_sample_book_reviews()
preprocessor = TextPreprocessor()
reviews = preprocessor.preprocess_dataframe(reviews)

# 预测
X = vectorizer.transform(reviews['Cleaned_Review'])
predictions = model.predict(X)

# 评估
from sklearn.metrics import accuracy_score
actual = (reviews['Rating'] >= 4).astype(int)
accuracy = accuracy_score(actual, predictions)
print(f"Accuracy: {accuracy:.2%}")
```

---

## 📊 预期结果

### 场景 1: 模型迁移效果好 (Accuracy > 80%)
✅ **说明**：通用情感词（good, bad, excellent, terrible）在食品和书评中都有效  
💡 **建议**：模型可以直接使用，但建议微调以提高准确度

### 场景 2: 模型迁移效果中等 (60-80%)
⚠️ **说明**：部分词汇迁移，但域特定词汇造成误差  
💡 **建议**：使用迁移学习，在书评数据上fine-tune

### 场景 3: 模型迁移效果差 (< 60%)
❌ **说明**：食品词汇与书评词汇差异太大  
💡 **建议**：重新训练书评专用模型

---

## 🌐 真实数据爬取

### Goodreads爬虫

```python
from book_review_scraper import BookReviewScraper

scraper = BookReviewScraper()

# 爬取Goodreads书评
book_url = "https://www.goodreads.com/book/show/YOUR_BOOK_ID"
reviews = scraper.scrape_goodreads_book(book_url, max_reviews=50)
```

### Amazon Books爬虫

```python
# 使用ASIN (Amazon product ID)
asin = "B08XYZABC1"  # 示例
reviews = scraper.scrape_amazon_books(asin, max_reviews=50)
```

**注意**：
- 爬虫可能需要根据网站更新调整selector
- 添加延时避免被封IP：`sleep(2)`
- 尊重网站的robots.txt

---

## 💡 改进建议

### 1. 数据增强
- 收集500-1000条真实书评
- 平衡正负样本
- 多样化书籍类型（小说、非虚构、技术书）

### 2. 模型优化
**Option A - Fine-tuning**：
```python
# 使用食品模型参数作为初始值
# 在书评数据上继续训练
model.fit(X_book_train, y_book_train)
```

**Option B - Ensemble**：
```python
# 组合食品模型 + 书评模型
predictions = (food_model.predict_proba(X) + book_model.predict_proba(X)) / 2
```

**Option C - Transfer Learning with BERT**：
```python
from transformers import BertForSequenceClassification
# 使用pre-trained BERT，在书评上fine-tune
```

### 3. 特征工程
添加书评特定特征：
- 包含"character", "plot", "writing"等词的频率
- 评论长度（详细评论通常=强烈情感）
- 比较词（"better than", "worse than"）

---

## 🔧 常见问题

**Q: 为什么需要保存vectorizer？**  
A: 模型需要将文本转换为与训练时**相同维度**的向量。没有相同的vectorizer，新文本无法正确编码。

**Q: 我的模型在书评上效果很差，怎么办？**  
A: 这是正常的！食品和书籍是完全不同的领域。建议：
1. 收集书评训练数据
2. Fine-tune现有模型
3. 或训练新的书评专用模型

**Q: 中文书评可以吗？**  
A: **不行**。你的模型只支持英文，因为：
- WordNetLemmatizer只支持英文
- 中文需要jieba分词
- stop words列表不同

如果要支持中文，需要完全重写预处理pipeline。

**Q: 我没有internet怎么测试？**  
A: notebook里有20条样本书评，足够测试模型功能。等有网络后再爬取真实数据。

---

## 📈 下一步

1. ✅ 运行测试notebook，查看初步结果
2. ✅ 分析哪些类型的书评被误分类
3. ✅ 爬取50-100条真实Goodreads书评
4. ✅ 在真实数据上重新评估
5. ✅ 根据结果决定是否需要fine-tune
6. ✅ 考虑升级到BERT等modern方法

---

## 📚 技术栈

- **NLP**: NLTK, scikit-learn TfidfVectorizer
- **ML**: Logistic Regression (from your original project)
- **Web Scraping**: BeautifulSoup, requests
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn

---

## 🎓 学习要点

这个项目展示了：
1. ✅ **Domain Transfer**: 跨领域模型迁移的挑战
2. ✅ **Model Deployment**: 如何在新数据上部署训练好的模型
3. ✅ **Pipeline Consistency**: 预处理pipeline必须完全一致
4. ✅ **Web Scraping**: 真实数据获取
5. ✅ **Model Evaluation**: 全面的性能分析

非常适合作为**resume项目**展示你的端到端ML能力！

---

## ✨ Resume子弹点参考

```
• Evaluated Amazon food review sentiment model on book review domain, achieving XX% 
  accuracy and identifying key vocabulary gaps through TF-IDF feature analysis

• Built automated web scraper using BeautifulSoup to collect 500+ Goodreads reviews, 
  enabling cross-domain model testing and transfer learning evaluation

• Analyzed domain transfer performance through confusion matrix and misclassification 
  patterns, providing data-driven recommendations for model fine-tuning
```

---

Good luck! 🚀

**Questions?** Review the notebook comments - 每个cell都有详细解释！
