import json
import pandas as pd
import time
from typing import Dict, List, Tuple
from datetime import datetime
import requests
import os
import re
from tqdm import tqdm

debug = False

class NewsAnalyzer:
    def __init__(self):
        self.market_impact_df = pd.DataFrame(columns=['date', 'source'])
        self.company_mentions_df = pd.DataFrame(columns=['date', 'source'])
        self.all_companies = set()
        self.valid_tickers = self._load_valid_tickers()
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.checkpoint_file = "checkpoint.txt"
        
    def _load_valid_tickers(self) -> set:
        """Load valid tickers from company_info.csv"""
        try:
            company_info = pd.read_csv('company_info.csv')
            return set(company_info['tic'])
        except Exception as e:
            print(f"Error loading company_info.csv: {str(e)}")
            return set()

    def _process_tickers(self, tickers: List[str], company_names: List[str]) -> Tuple[List[str], List[str]]:
        """Process and validate tickers against company_info.csv"""
        valid_pairs = []
        for ticker, name in zip(tickers, company_names):
            base_ticker = ticker.split()[0]
            if base_ticker in self.valid_tickers:
                valid_pairs.append((base_ticker, name))
        
        if not valid_pairs:
            return [], []
            
        return zip(*valid_pairs)

    def analyze_with_deepseek(self, title: str, content: str, tickers: List[str], company_names: List[str]) -> Dict[str, int]:
        """Analyze market impact based on news content using deepseek
        Impact score range:
        2: Strongly Positive (major business expansion, significant financial growth)
        1: Positive (moderate positive developments)
        0: Neutral (purely factual news, rarely used)
        -1: Negative (moderate negative developments)
        -2: Strongly Negative (major business setbacks, significant financial decline)
        """
        try:
            # Combine title and content
            text = f"{title}. {content}"
            
            # Create mapping between company names and tickers
            name_to_ticker = dict(zip(company_names, tickers))
            
            # Prepare prompt for deepseek
            prompt = f"""You are a financial news analyst. Analyze the following news and provide sentiment scores for each mentioned company.
Do not think deeply, just provide direct answers. 减少思考，快速得出结论。
Do not remember context from previous messages.
Be precise and conservative in your analysis.

News content:
{text}

Companies to analyze: {', '.join(company_names)}

For each company, provide ONLY a number (-2 to 2) indicating the market sentiment score:
2: Strongly Positive (major business expansion, significant financial growth)
1: Positive (moderate positive developments)
0: Neutral (purely factual news, rarely used)
-1: Negative (moderate negative developments)
-2: Strongly Negative (major business setbacks, significant financial decline)

Only in very rare cases, provide a score of 0. You must try to provide a non-zero score.

Positive Examples:
moderate growth
launches new product line
positive earnings report
expands operations
merger announced
cost reduction
contracts secured
partnership announced
improves efficiency

Negative Examples:
missed revenue expectations
supply chain issues
increased costs
legal dispute initiated
product delay
downgraded outlook

请不要解释过程，不要深度思考，直接输出每家公司的评分，并且用英文回复，不要任何解释。
### Output format:
Company1: score
Company2: score
...

Example:
Exxon Mobil Corp: 1
Diamondback Energy Inc: -1
ProPetro Holding Corp: 2

Attention: Never respond like this:
Company: XX Co
Score: 1

Provide scores for ALL Companies in Companies to analyze."""

            # Call ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "deepseek-r1:8b",
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Remove thinking part from response
                response_text = result['response']
                if '<think>' in response_text and '</think>' in response_text:
                    response_text = response_text.split('</think>')[-1].strip()
                
                impact_scores = {}

                # try to avoid company and score format
                temp_companies = None
                for line in response_text.split('\n'):
                    if ':' in line:
                        item1, item2 = line.split(':', 1)
                        if re.findall(r'company', item1.lower()) and type(item2) != int:
                            temp_companies = item2
                        if temp_companies is not None and re.findall(r'score', item2.lower()) and item2 in ("-2","-1","0","1","2"):
                            response_text += f"\n{temp_companies}:{item2}"
                            temp_companies = None
                
                # Parse the response and convert company names back to tickers
                for line in response_text.split('\n'):
                    if ':' in line:
                        company, score = line.split(':', 1)
                        company = company.strip().replace(".","").replace("- **","").replace("**","")
                        
                        if company in name_to_ticker:
                            try:
                                score = int(score.strip())
                                if score in [-2, -1, 0, 1, 2]:
                                    impact_scores[name_to_ticker[company]] = score
                                else:
                                    print(f"Warning: Invalid score {score} for company {company}")
                                    impact_scores[name_to_ticker[company]] = 0
                            except ValueError:
                                print(f"Warning: Could not parse score for company {company}")
                                impact_scores[name_to_ticker[company]] = 0
                        # else:
                            # print(f"Warning: Unknown company name in response: {company}")

                

                return impact_scores
            else:
                print(f"Error calling ollama API: {response.status_code}")
                return {}
            
        except Exception as e:
            print(f"Error in deepseek analysis: {str(e)}")
            return {}

    def save_results(self, append=False):
        """Save results to CSV files"""
        try:
            self.market_impact_df = self.market_impact_df.fillna(0)
            self.company_mentions_df = self.company_mentions_df.fillna(0)
            
            # Check if files exist
            market_impact_exists = os.path.exists('market_impact.csv')
            company_mentions_exists = os.path.exists('company_mentions.csv')
            
            # If file doesn't exist, use 'w' mode to create new file
            # If file exists and append=True, use 'a' mode to append
            # If file exists and append=False, use 'w' mode to overwrite
            market_impact_mode = 'a' if (market_impact_exists and append) else 'w'
            company_mentions_mode = 'a' if (company_mentions_exists and append) else 'w'
            
            # Only write header when file doesn't exist or append=False
            market_impact_header = not (market_impact_exists and append)
            company_mentions_header = not (company_mentions_exists and append)
            
            self.market_impact_df.to_csv('market_impact.csv', index=False, mode=market_impact_mode, header=market_impact_header)
            self.company_mentions_df.to_csv('company_mentions.csv', index=False, mode=company_mentions_mode, header=company_mentions_header)
            
            if not append:
                print("Results saved to CSV files")
            else:
                print(f"Appended {len(self.market_impact_df)} rows to CSV files")
                
            # Clear DataFrames after saving
            self.market_impact_df = pd.DataFrame(columns=['date', 'source'])
            self.company_mentions_df = pd.DataFrame(columns=['date', 'source'])
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def _load_checkpoint(self) -> int:
        """Load the last processed news count from checkpoint file"""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    return int(f.read().strip())
            return 0
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return 0
            
    def _save_checkpoint(self, count: int):
        """Save the current processed news count to checkpoint file"""
        try:
            with open(self.checkpoint_file, 'w') as f:
                f.write(str(count))
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    def process_news(self, news_list: List[Dict]):
        """Process news list"""
        total_news = len(news_list)
        processed_count = 0
        error_count = 0
        save_interval = 50
        
        # Load checkpoint
        start_from = self._load_checkpoint()
        
        # Initialize empty DataFrames
        self.market_impact_df = pd.DataFrame(columns=['date', 'source'])
        self.company_mentions_df = pd.DataFrame(columns=['date', 'source'])
        
        # Create progress bar
        pbar = tqdm(total=total_news, initial=start_from, desc="Processing news")
        
        for i, news in enumerate(news_list):
            # Skip already processed news
            if i < start_from:
                continue
                
            try:
                date = news['date']
                source = news['source']
                title = news['title']
                
                # Process and validate tickers
                tickers, company_names = self._process_tickers(
                    news['tickers'],
                    news['company_names']
                )
                
                if not tickers:
                    pbar.update(1)
                    continue
                
                # Update all companies set with valid tickers
                self.all_companies.update(tickers)
                
                # Analyze with deepseek
                market_impact = self.analyze_with_deepseek(
                    title,
                    news['content'],
                    tickers,
                    company_names
                )
                
                if not market_impact:
                    pbar.update(1)
                    continue
                
                # Create mentions dictionary (1 for mentioned companies)
                mentions = {ticker: 1 for ticker in tickers}
                
                # Update DataFrames
                self._update_dataframes(date, source, market_impact, mentions)
                
                processed_count += 1
                
                # Save results and checkpoint every save_interval news items
                if processed_count % save_interval == 0:
                    self.save_results(append=(processed_count > save_interval))
                    self._save_checkpoint(i + 1)  # Save checkpoint after current item
                
                pbar.update(1)
                
            except Exception as e:
                error_count += 1
                pbar.update(1)
                continue
        
        # Save any remaining results
        if len(self.market_impact_df) > 0:
            self.save_results(append=(processed_count > save_interval))
            self._save_checkpoint(total_news)  # Save final checkpoint
        
        pbar.close()
        
        print(f"\nProcessing completed:")
        print(f"Total news items: {total_news}")
        print(f"Successfully processed: {processed_count}")
        print(f"Errors encountered: {error_count}")

    def _update_dataframes(self, date, source, market_impact, mentions):
        """Update DataFrames with market impact and mentions"""
        # Convert valid_tickers set to list for DataFrame columns
        valid_tickers_list = list(self.valid_tickers)
        
        if not market_impact:
            return None
        
        # Create base row with all zeros
        base_row = {
            'date': date,
            'source': source,
            **{ticker: 0 for ticker in valid_tickers_list}
        }
        
        # Create new rows for both DataFrames
        new_impact_row = pd.DataFrame([base_row])
        new_mention_row = pd.DataFrame([base_row])
        
        # Concatenate with existing DataFrames
        self.market_impact_df = pd.concat([self.market_impact_df, new_impact_row])
        self.company_mentions_df = pd.concat([self.company_mentions_df, new_mention_row])
        
        # Get the last row index
        last_row_idx = len(self.market_impact_df) - 1
        
        # Update values for both DataFrames
        for company in valid_tickers_list:
            if company in market_impact:
                self.market_impact_df.iloc[last_row_idx, self.market_impact_df.columns.get_loc(company)] = market_impact[company]
            if company in mentions:
                self.company_mentions_df.iloc[last_row_idx, self.company_mentions_df.columns.get_loc(company)] = mentions[company]

def main():
    analyzer = NewsAnalyzer()
    
    # Load news from JSON files
    try:
        # Read both JSON files
        with open('AIDF_FT5005.AIDF_FT5005_oil_w_content.json', 'r', encoding='utf-8') as f:
            news_with_content = json.load(f)
        
        # Combine news from both files
        news_list = news_with_content
        
        # Process news
        analyzer.process_news(news_list)
        analyzer.save_results(append=True)
        
    except FileNotFoundError as e:
        print(f"Error: JSON file not found: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {str(e)}")
    except Exception as e:
        print(f"Error loading news data: {str(e)}")

if __name__ == "__main__":
    main() 