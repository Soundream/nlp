import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import Dict, List, Tuple
from gensim import corpora, models
from gensim.models import CoherenceModel
import numpy as np
from tqdm import tqdm
import json

class NewsLDAAnalyzer:
    def __init__(self, num_topics: int = 5):
        self.num_topics = num_topics
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.dictionary = None
        self.lda_model = None
        self.results_df = pd.DataFrame()
        self.valid_tickers = self._load_valid_tickers()
        
    def _load_valid_tickers(self) -> set:
        """Load valid tickers from company_info.csv"""
        try:
            company_info = pd.read_csv('company_info.csv')
            return set(company_info['tic'])
        except Exception as e:
            print(f"Error loading company_info.csv: {str(e)}")
            return set()
            
    def _process_tickers(self, tickers: List[str]) -> List[str]:
        """Process and validate tickers against company_info.csv"""
        valid_tickers = []
        for ticker in tickers:
            base_ticker = ticker.split()[0]
            if base_ticker in self.valid_tickers:
                valid_tickers.append(base_ticker)
        return valid_tickers
        
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert NLTK POS tags to WordNet POS tags"""
        if treebank_tag.startswith('J'):
            return 'a'  # adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # verb
        elif treebank_tag.startswith('N'):
            return 'n'  # noun
        elif treebank_tag.startswith('R'):
            return 'r'  # adverb
        else:
            return 'n'  # default to noun
            
    def _lemmatize_text(self, text: str) -> List[str]:
        """Perform lemmatization on text"""
        tokens = word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(tokens)
        lemmatized_words = []
        for word, tag in pos_tags:
            if word not in self.stop_words and word.isalpha():
                pos = self._get_wordnet_pos(tag)
                lemmatized_word = self.lemmatizer.lemmatize(word, pos)
                lemmatized_words.append(lemmatized_word)
        return lemmatized_words
        
    def _prepare_corpus(self, documents: List[str]) -> Tuple[List[List[str]], corpora.Dictionary]:
        """Prepare corpus for LDA"""
        processed_docs = [self._lemmatize_text(doc) for doc in documents]
        dictionary = corpora.Dictionary(processed_docs)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        return corpus, dictionary
        
    def train_lda(self, documents: List[str]):
        """Train LDA model"""
        corpus, dictionary = self._prepare_corpus(documents)
        self.dictionary = dictionary
        
        self.lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=[self._lemmatize_text(doc) for doc in documents],
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        print(f"Coherence Score: {coherence_score}")
        
    def get_topic_words(self, num_words: int = 20) -> Dict[int, List[Tuple[str, float]]]:
        """Get top words and their probabilities for each topic"""
        if not self.lda_model:
            raise ValueError("LDA model not trained yet")
            
        topic_words = {}
        for topic_id in range(self.num_topics):
            words = self.lda_model.show_topic(topic_id, num_words)
            topic_words[topic_id] = words
            
        return topic_words
        
    def process_news(self, news_list: List[Dict]) -> pd.DataFrame:
        """Process news list and assign topics"""
        # 创建结果DataFrame
        columns = ['index', 'date', 'source'] + [f'topic_{i}_prob' for i in range(self.num_topics)]
        results_df = pd.DataFrame(columns=columns)
        
        # 先筛选出有效的新闻
        valid_news = []
        for news in news_list:
            tickers = news.get('tickers', [])
            valid_tickers = self._process_tickers(tickers)
            if valid_tickers:
                valid_news.append(news)
        
        # 提取有效新闻的内容用于训练
        documents = [news['content'] for news in valid_news]
        
        # 训练 LDA 模型
        print("开始训练 LDA 模型...")
        self.train_lda(documents)
        
        # 处理新闻并分配主题
        for index, news in tqdm(enumerate(valid_news), total=len(valid_news), desc="Processing news"):
            try:
                # 获取文档主题分布
                bow = self.dictionary.doc2bow(self._lemmatize_text(news['content']))
                topic_dist = self.lda_model.get_document_topics(bow)
                
                # 创建新行，初始化所有主题概率为0
                new_row = {
                    'index': index,
                    'date': news['date'],
                    'source': news['source']
                }
                for i in range(self.num_topics):
                    new_row[f'topic_{i}_prob'] = 0.0
                
                # 更新实际的主题概率
                for topic_id, prob in topic_dist:
                    new_row[f'topic_{topic_id}_prob'] = prob
                
                # 添加到结果DataFrame
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                
            except Exception as e:
                print(f"Error processing news at index {index}: {str(e)}")
                continue
                
        return results_df
        
    def save_results(self, results_df: pd.DataFrame, output_file: str = 'lda_results.csv'):
        """Save results to CSV file"""
        results_df.to_csv(output_file, index=False)
        
    def save_topic_words(self, topic_words: Dict[int, List[Tuple[str, float]]], output_file: str = 'topic_words.txt'):
        """Save topic words to text file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for topic_id, words in topic_words.items():
                f.write(f"Topic {topic_id}:\n")
                for word, prob in words:
                    f.write(f"{word}: {prob:.4f}\n")
                f.write("\n")

def main():
    # 下载必要的 NLTK 数据
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    # 读取新闻数据
    try:
        with open('AIDF_FT5005.AIDF_FT5005_oil_w_content.json', 'r', encoding='utf-8') as f:
            news_list = json.load(f)
    except FileNotFoundError:
        print("找不到新闻数据文件")
        return
    
    # 初始化分析器
    analyzer = NewsLDAAnalyzer(num_topics=5)
    
    # 处理新闻并保存结果
    print("\n开始处理新闻...")
    results_df = analyzer.process_news(news_list)
    
    # 获取主题词汇
    topic_words = analyzer.get_topic_words(num_words=20)
    
    # 保存结果
    analyzer.save_results(results_df, 'lda_results.csv')
    analyzer.save_topic_words(topic_words, 'topic_words.txt')
    print("\n分析完成，结果已保存")

if __name__ == "__main__":
    main() 