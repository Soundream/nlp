import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import Dict, List, Tuple
import re
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

class NewsDictionaryAnalyzer:
    def __init__(self, custom_dict: Dict[str, List[str]]):
        self.custom_dict = custom_dict
        self.lemmatizer = WordNetLemmatizer()
        # Load financial stopwords
        self.stop_words = set(stopwords.words('english'))
        # Initialize DataFrame with all required columns
        self.results_df = pd.DataFrame(columns=['index', 'date', 'source'] + list(custom_dict.keys()))
        # Load valid tickers
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
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Get POS tags
        pos_tags = nltk.pos_tag(tokens)
        # Lemmatize
        lemmatized_words = []
        for word, tag in pos_tags:
            if word not in self.stop_words and word.isalpha():
                pos = self._get_wordnet_pos(tag)
                lemmatized_word = self.lemmatizer.lemmatize(word, pos)
                lemmatized_words.append(lemmatized_word)
        return lemmatized_words
        
    def _lemmatize_dict_terms(self) -> Dict[str, List[str]]:
        """Lemmatize dictionary terms"""
        lemmatized_dict = {}
        for category, terms in self.custom_dict.items():
            lemmatized_terms = []
            for term in terms:
                lemmatized_words = self._lemmatize_text(term)
                lemmatized_terms.append(' '.join(lemmatized_words))
            lemmatized_dict[category] = lemmatized_terms
        return lemmatized_dict
        
    def _calculate_tf(self, text: str, lemmatized_dict: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate term frequency for each category"""
        lemmatized_text = self._lemmatize_text(text)
        tf_scores = {}
        
        for category, terms in lemmatized_dict.items():
            count = 0
            for term in terms:
                term_words = term.split()
                # Check if term exists in text
                for i in range(len(lemmatized_text) - len(term_words) + 1):
                    if lemmatized_text[i:i+len(term_words)] == term_words:
                        count += 1
                        break  # 找到一个匹配就跳出内层循环
            tf_scores[category] = count
            
        return tf_scores
        
    def process_news(self, news_list: List[Dict]):
        """Process news list"""
        lemmatized_dict = self._lemmatize_dict_terms()
        
        for index, news in tqdm(enumerate(news_list), total=len(news_list), desc="Processing news"):
            try:
                # Get basic information
                date = news['date']
                source = news['source']
                content = news['content']
                tickers = news['tickers']
                
                # Validate tickers
                valid_tickers = self._process_tickers(tickers)
                if not valid_tickers:
                    continue
                
                # Calculate TF
                tf_scores = self._calculate_tf(content, lemmatized_dict)
                
                # Create new row
                new_row = {
                    'index': index,
                    'date': date,
                    'source': source,
                    **tf_scores
                }
                
                # Add to DataFrame
                new_df = pd.DataFrame([new_row])
                self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)
                
            except Exception as e:
                print(f"Error processing news at index {index}: {str(e)}")
                continue
                
    def save_results(self, output_file: str = 'dictionary_analysis_results.csv'):
        """Save results to CSV file"""
        self.results_df.to_csv(output_file, index=False)
        
def main():
    # Example dictionary
    custom_dict = {
        "supply_side": [
            "rig count",
            "shale",
            "OPEC",
            "inventory build",
            "production cut",
            "refinery outage",
            "crude output",
            "drilling activity",
            "offshore",
            "supply disruption",
            "barrel output",
            "oil field",
            "pipeline capacity",
            "refinery throughput",
            "exploration project",
            "well shutdown",
            "capacity expansion",
            "spare capacity",
            "supply glut",
            "maintenance shutdown",
            "new well discovery",
            "shale gas",
            "unconventional reserve",
            "tight oil",
            "enhanced recovery",
            "supply chain",
            "production quota",
            "drilling permit",
            "oil sands",
            "fracking",
            "reservoir pressure",
            "well completion",
            "production decline",
            "field development",
            "extraction rate",
            "crude quality",
            "storage capacity",
            "export terminal",
            "tanker fleet",
            "loading facility"
        ],
        "demand_side": [
            "fuel demand",
            "jet fuel",
            "gasoline consumption",
            "refinery utilization",
            "industrial activity",
            "economic slowdown",
            "freight traffic",
            "seasonal demand",
            "consumption outlook",
            "demand recovery",
            "aviation fuel",
            "vehicle mileage",
            "diesel usage",
            "holiday travel",
            "transportation index",
            "manufacturing output",
            "public mobility",
            "urban traffic",
            "retail fuel sales",
            "economic reopening",
            "energy intensity",
            "fuel efficiency",
            "consumer behavior",
            "travel pattern",
            "commuting trend",
            "logistics demand",
            "shipping volume",
            "air travel",
            "road transport",
            "fuel switching",
            "energy transition",
            "electric vehicle",
            "hybrid car",
            "public transport",
            "fuel subsidy",
            "energy policy",
            "urbanization",
            "population growth",
            "economic growth",
            "industrial output"
        ],
        "financial_risk": [
            "hedging loss",
            "impairment",
            "capex cut",
            "cash burn",
            "leverage",
            "debt maturity",
            "bankruptcy",
            "cost overrun",
            "default risk",
            "liquidity crunch",
            "debt restructuring",
            "credit downgrade",
            "loan covenant",
            "asset writedown",
            "financial distress",
            "equity dilution",
            "interest coverage",
            "bond yield",
            "dividend suspension",
            "profit warning",
            "market volatility",
            "currency risk",
            "commodity price",
            "inflation pressure",
            "interest rate",
            "credit rating",
            "debt ratio",
            "cash flow",
            "working capital",
            "capital expenditure",
            "revenue decline",
            "margin pressure",
            "cost inflation",
            "tax liability",
            "pension obligation",
            "derivative exposure",
            "counterparty risk",
            "refinancing risk",
            "credit spread",
            "equity valuation"
        ],
        "geopolitical_risk": [
            "sanctions",
            "military conflict",
            "embargo",
            "export ban",
            "trade war",
            "political instability",
            "supply chain disruption",
            "strategic reserve release",
            "Middle East tension",
            "nationalization",
            "border dispute",
            "diplomatic standoff",
            "regime change",
            "energy weaponization",
            "territorial conflict",
            "naval blockade",
            "foreign intervention",
            "geopolitical standoff",
            "proxy war",
            "military escalation",
            "political unrest",
            "civil war",
            "terrorist attack",
            "cyber attack",
            "espionage",
            "intelligence operation",
            "military exercise",
            "arms race",
            "nuclear program",
            "missile test",
            "border tension",
            "territorial claim",
            "maritime dispute",
            "airspace violation",
            "military buildup",
            "defense spending",
            "security threat",
            "intelligence sharing",
            "military alliance",
            "security pact"
        ],
        "esg_regulation": [
            "carbon tax",
            "emissions cap",
            "net zero",
            "climate policy",
            "sustainability goal",
            "renewable mandate",
            "green bond",
            "ESG disclosure",
            "regulatory fine",
            "climate litigation",
            "environmental audit",
            "emission trading",
            "green investment",
            "transition risk",
            "ESG rating",
            "decarbonization plan",
            "regulatory compliance",
            "climate risk disclosure",
            "fossil fuel divestment",
            "corporate responsibility",
            "carbon footprint",
            "renewable energy",
            "energy efficiency",
            "waste management",
            "water conservation",
            "biodiversity",
            "environmental impact",
            "social responsibility",
            "governance structure",
            "stakeholder engagement",
            "sustainability report",
            "climate action",
            "green technology",
            "circular economy",
            "sustainable development",
            "environmental standard",
            "carbon offset",
            "renewable certificate",
            "sustainability metric",
            "ESG performance"
        ],
        "natural_factor": [
            "hurricane",
            "heat wave",
            "cold snap",
            "blizzard",
            "drought",
            "wildfire",
            "El Niño",
            "La Niña",
            "natural gas supply",
            "gas storage level",
            "freeze-off",
            "snowstorm",
            "pipeline freeze",
            "weather forecast",
            "arctic blast",
            "temperature anomaly",
            "power outage",
            "gas-fired generation",
            "LNG supply",
            "storm surge",
            "tropical storm",
            "flooding",
            "earthquake",
            "tsunami",
            "volcanic activity",
            "seasonal variation",
            "climate pattern",
            "ocean current",
            "atmospheric pressure",
            "wind pattern",
            "precipitation level",
            "humidity level",
            "temperature extreme",
            "weather system",
            "climate change",
            "sea level",
            "glacier melt",
            "permafrost thaw",
            "ocean temperature",
            "atmospheric condition"
        ]
    }
    
    # Create analyzer
    analyzer = NewsDictionaryAnalyzer(custom_dict)
    
    # Load news data
    try:
        with open('AIDF_FT5005.AIDF_FT5005_oil_w_content.json', 'r', encoding='utf-8') as f:
            news_list = json.load(f)
            
        # Process news
        analyzer.process_news(news_list)
        
        # Save results
        analyzer.save_results()
        
    except FileNotFoundError as e:
        print(f"Error: JSON file not found: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {str(e)}")
    except Exception as e:
        print(f"Error loading news data: {str(e)}")

if __name__ == "__main__":
    main() 