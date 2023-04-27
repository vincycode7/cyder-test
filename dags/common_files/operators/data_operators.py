import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import spacy, os
import hashlib, requests, re, spacy
from typing import List, Optional, Tuple
import nltk
from collections import Counter

def load_or_download_models():
    spacy_model_name = 'en_core_web_sm'
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        spacy.cli.download(spacy_model_name)
        try:
            nlp = spacy.load(spacy_model_name)
        except Exception as e:
            raise Exception(f"Failed Process Due to {e}")
    output_dir="outputs/data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  
    
class DataAnalyzer:
    def __init__(self, link_to_csv: str, pii_to_mask=None, pii_to_remove=None, output_file=None, output_dir="outputs/data/"):
        """
        A class for analyzing data.

        Parameters:
        data (List[dict]): The data to be analyzed.
        """
        self.data = pd.read_csv(link_to_csv)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)   
                     
        self.output_file = output_dir+"output_analysis_"+link_to_csv.split("/")[-1]
        self.pii_to_mask = pii_to_mask or ['userId']
        self.pii_to_remove = pii_to_remove or ['userId','ip']
        self.metadata_data = ['metadata.name', 'metadata.content']
        spacy_model_name = 'en_core_web_sm'
        self.nlp = spacy.load(spacy_model_name)
        
    def gentle_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        A gentle data preprocessing method that replaces NaN values with empty strings and strips whitespace from the remaining values.

        Args:
            data (pd.DataFrame): The input data to preprocess.

        Returns:
            pd.DataFrame: The preprocessed data.
        """
        return data.fillna("").apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    def drop_pii_col(self, cols: Optional[List[str]] = None) -> None:
        """
        Removes personally identifiable information (PII) from the data.

        Parameters:
        cols (Optional[List[str]]): The list of column names to remove. If not provided,
        the default list of PII columns ['userId', 'ip'] will be used.
        """
        cols = cols or self.pii_to_remove or ['userId']
        missing_cols = set(cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")

        self.data = self.data.drop(columns=[col for col in cols if col in self.data.columns], errors='ignore')
    
    def mask_pii(self, cols: Optional[List[str]] = None) -> None:
        """
        Hashes the specified columns in the data using the SHA256 algorithm.

        Parameters:
        cols (Optional[List[str]]): The list of column names to hash. Defaults to ['userId'].
        """
        cols = cols or self.pii_to_mask or ['userId']
        missing_cols = set(cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")

        for col in cols:
            self.data[col] = self.data[col].apply(lambda x: hashlib.sha256(str(x).encode('utf-8')).hexdigest())

    def remove_pii_from_metadata(self, cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Removes personally identifiable information (PII) from the metadata of a pandas DataFrame.

        Args:
            cols (Optional[List[str]]): A list of column names to remove PII from. Default is ['userId'].

        Returns:
            pd.DataFrame: The modified DataFrame with PII removed.

        Raises:
            ValueError: If any column names in `cols` are not present in the DataFrame.
        """
        cols = cols or self.metadata_data or ['metadata.name', 'metadata.content']
        missing_cols = set(cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")

        for col in cols:
            if col in ['metadata.name', 'metadata.content']:
              self.data[col] = self.data[col].apply(lambda x: re.sub(r"\b\d{9}\b|\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "", str(x)))
        return self.data

    def anonymize_user(self):
        self.mask_pii()
        self.drop_pii_col()
        self.remove_pii_from_metadata()

    def extract_interests_and_demo(self) -> pd.DataFrame:
        """
        Extracts interests from metadata.

        Returns:
            pd.DataFrame: The modified DataFrame with interests extracted.
        """
        if 'metadata.content' not in self.data.columns:
            raise ValueError("Column 'metadata.content' not found in DataFrame")
            
        interests, income_range, is_finance_related = zip(*self.data['metadata.content'].apply(self.extract_interests_and_demo_from_text))
        self.data['interests'] = interests
        # self.data['age_group'] = age_group
        self.data['income_range'] = income_range
        self.data['is_finance_related'] = is_finance_related

        return self.data

    def extract_interests_and_demo_from_text(self, text: str):
        doc = self.nlp(text)
        interests = self.extract_interests_from_doc(doc)
        # age_group = self.extract_age_group_from_doc(doc)
        income_range = self.extract_income_range_from_doc(doc)
        is_finance_related = self.extract_is_finance_related_from_doc(text)
        return interests, income_range, is_finance_related

    def extract_interests_from_doc(self, doc: List[str]) -> List[str]:
        """
        Extracts interests from a given text using spaCy.

        Args:
            text (str): The text to extract interests from.

        Returns:
            List[str]: A list of extracted interests with their associated named entities.
        """
        interests = []
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG", "GPE"]:
                interests.append(ent.text)
        return ' '.join(interests)

    def extract_age_group_from_doc(self, doc: List[str]) -> str:
        """
        Extracts age group from a given text.

        Args:
            text (str): The text to extract age group from.

        Returns:
            str: The extracted age group.
        """
        age_group = ""
        for ent in doc.ents:
            if ent.label_ == "AGE":
                age_group = ent.text
        return age_group

    def extract_income_range_from_doc(self, doc: List[str]) -> str:
        """
        Extracts income range from a given text.

        Args:
            text (str): The text to extract income range from.

        Returns:
            str: The extracted income range.
        """
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                # clean the money value
                income_str = ent.text.replace(",", "").replace("$", "")
                # convert the cleaned value to float
                try:
                    income = float(income_str)
                except ValueError:
                    return "undefined"
                # map the income to a text range
                return self.map_income_to_range(income)
        # no money entity found
        return "undefined"

    def map_income_to_range(self,income: float) -> str:
        """
        Maps income value to a text income range.

        Args:
            income (float): The income value to map.

        Returns:
            str: The mapped income range.
        """
        if income < 0:
            return "Undefined"
        elif income < 100000:
            return "Below $100,000"
        else:
            return "Above $100,000"

    def extract_is_finance_related_from_doc(self, text: str) -> bool:
        """
        Determines if the given text is finance related.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if the text is finance related, False otherwise.
        """
        finance_keywords = ['finance', 'financial', 'invest', 'investment', 'stock',     
                            'portfolio', 'mutual fund', 'bond', 'wealth management',     
                            'hedge fund', 'equity', 'market', 'capital', 'fund',     
                            'asset', 'commodity', 'derivatives', 'trading', 'brokerage',     
                            'valuation', 'risk management', 'options', 'futures',     
                            'real estate investment trust', 'private equity',     
                            'venture capital', 'insurance', 'retirement planning',     
                            'taxation', 'credit', 'loan', 'banking', 'accounting',     
                            'audit', 'financial planning', 'economic', 'macroeconomics',     
                            'microeconomics', 'monetary policy', 'fiscal policy',     
                            'budget', 'debt', 'interest rate', 'inflation',     
                            'exchange rate', 'foreign exchange', 'cryptocurrency',     
                            'blockchain', 'digital currency', 'initial coin offering',     
                            'smart contract', 'decentralized finance', 'financial technology',     
                            'payment system', 'credit card', 'debit card',     
                            'mobile payment', 'e-commerce', 'online payment',     
                            'crowdfunding', 'peer-to-peer lending', 'robo-advisor',     
                            'artificial intelligence in finance',     
                            'machine learning in finance',     
                            'quantitative finance'
                            ]

        tokens = nltk.word_tokenize(text.lower())
        if set(finance_keywords) & set(tokens):
            return True
        return False

    def derive_attributes(self) -> None:
        """
        Derives demographic and interest-based attributes from the data.
        """
        self.extract_interests_and_demo()

    def save_data(self, output_file: str=None) -> None:
        """
        Saves the analyzed data to a CSV file.

        Args:
            output_file (str): The file path of the output CSV file.
        """
        output_file = output_file or self.output_file
        self.data.to_csv(output_file, index=False)

    def analyze_data(self):
        """
        Analyzes the data and returns a summary of insights.
        """
        self.data = self.gentle_preprocess(self.data)
        self.anonymize_user()
        self.derive_attributes()
        self.save_data()