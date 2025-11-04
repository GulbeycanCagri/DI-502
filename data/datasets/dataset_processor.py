import os
import re
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import google.generativeai as genai
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from newspaper import Article, Config
from tqdm import tqdm


import sec_parser as sp
from sec_parser.exceptions import SecParserError
from sec_downloader import Downloader
from sec_parser import TreeBuilder
from sec_parser.semantic_elements import TextElement

class FinancialNewsAnnotator:
    def __init__(self, api_key=None, model_name="gemini-flash-latest", sleep_time=1.5):
        # Find .env file
        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            try:
                script_dir = Path(__file__).resolve().parent
            except NameError:
                script_dir = Path.cwd()
            env_path = script_dir / ".env"

        load_dotenv(dotenv_path=env_path)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(f"Gemini API key not found. Looked in {env_path}")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.sleep_time = sleep_time

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        self.sec_headers = {
            "User-Agent": "Economind Project economind@example.com"
        }

        # Config for newspaper3k
        self.newspaper_config = Config()
        self.newspaper_config.browser_user_agent = self.headers["User-Agent"]
        self.newspaper_config.request_timeout = 20

        # --- NEW: sec-parser downloader ---
        self.sec_downloader = Downloader("Economind Team", "e274086@metu.edu.tr")
    
        # ----------------------------------

    # ==================================================================
    # SECTION 1: SEC EDGAR FUNCTIONS
    # ==================================================================

    def get_company_filings(self, cik, form_type="8-K", limit=10):
        # This function fetches the list of filings. No changes needed.
        cik10 = f"{int(cik):010d}"
        cik_int = int(cik)
        json_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        filings = []
        try:
            r = requests.get(json_url, headers=self.sec_headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            filings_data = data.get("filings", {}).get("recent", {})
            forms = filings_data.get("form", [])
            accession_nums = filings_data.get("accessionNumber", [])
            primary_docs = filings_data.get("primaryDocument", [])
            filing_dates = filings_data.get("filingDate", [])
            for i, form in enumerate(forms):
                if form == form_type:
                    acc, doc, date = accession_nums[i], primary_docs[i], filing_dates[i]
                    if not doc:
                        continue
                    filings.append(
                        {
                            "company": data["name"],
                            "cik": cik10,
                            "form": form_type,
                            "date": date,
                            "wrapper_url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc.replace('-', '')}/{doc}",
                        }
                    )
                if len(filings) >= limit:
                    break
            if not filings:
                print(f"ℹ️ [SEC] No {form_type} filings found for CIK {cik10}.")
            return filings
        except Exception as e:
            print(f"❌ [SEC] Request error for CIK {cik10}: {e}")
            return []

    def extract_exhibit_text(self, wrapper_url):
        # This function correctly finds EX-99.1 for 8-K filings. No changes needed.
        try:
            r = requests.get(wrapper_url, headers=self.headers, timeout=20)
            soup = BeautifulSoup(r.text, "html.parser")
            exhibit_link = None
            for row in soup.find_all("tr"):
                row_text = row.get_text(" ", strip=True)
                if re.search(r"EX-99\.1|Exhibit\s+99\.1", row_text, re.IGNORECASE):
                    link_tag = row.find("a", href=True)
                    if link_tag:
                        exhibit_link = link_tag["href"]
                        break
            if not exhibit_link:
                href_regex = re.compile(r"exh?[\-_]?99[\-_]?1", re.IGNORECASE)
                text_regex = re.compile(r"EX-99\.1|Exhibit\s+99\.1", re.IGNORECASE)
                for link_tag in soup.find_all("a", href=True):
                    if href_regex.search(link_tag.get("href", "")) or text_regex.search(
                        link_tag.get_text(strip=True)
                    ):
                        exhibit_link = link_tag["href"]
                        break
            if not exhibit_link:
                print(f"⚠️ [SEC] Exhibit 99.1 link not found in {wrapper_url}")
                return None
            exhibit_full_url = urljoin(wrapper_url, exhibit_link)
            ex_r = requests.get(exhibit_full_url, headers=self.headers, timeout=20)
            content_type = ex_r.headers.get("Content-Type", "").lower()
            is_html = "html" in content_type or exhibit_full_url.endswith(
                (".htm", ".html")
            )
            cleaned = ""
            if is_html:
                ex_soup = BeautifulSoup(ex_r.text, "html.parser")
                body = ex_soup.find("body") or ex_soup
                text_blocks = [
                    elem.get_text(" ", strip=True)
                    for elem in body.find_all(["p", "div", "span"], recursive=True)
                    if not elem.find("table") and elem.get_text(strip=True)
                ]
                cleaned = " ".join(text_blocks)
            else:
                cleaned = ex_r.text
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) < 250:
                print(
                    f"⚠️ [SEC] Extracted exhibit text too short from {exhibit_full_url}"
                )
                return None
            return cleaned
        except Exception as e:
            print(f"⚠️ [SEC] Failed to process exhibit from {wrapper_url}: {e}")
            return None

    # --- NEW METHOD FOR 10-K / 10-Q PARSING ---
    def extract_10k_report_items(self, filing_url):
        """
        Downloads the HTML from the given filing_url using 'requests'
        and parses it to find and extract the text of specific items.
        This version uses simple, robust regex patterns.
        """
        try:
            print(f"ℹ️ [sec-parser] Downloading and Parsing: {filing_url}")

            # 1. Download the HTML string using 'requests'
            r = requests.get(filing_url, headers=self.sec_headers, timeout=20)
            r.raise_for_status() 
            filing_html = r.text
            
            if not filing_html:
                print(f"⚠️ [requests] Downloaded HTML from {filing_url} is empty.")
                return None

            # 2. Parse the HTML (This is correct)
            parser = sp.Edgar10QParser()
            elements = parser.parse(filing_html)

            # 3. Build a semantic tree (This is correct)
            tree = TreeBuilder().build(elements)

            # 4. --- NEW, SIMPLER PATTERNS ---
            # We are now only looking for the item number at the
            # beginning of the line.
            target_item_patterns = [
                # Matches "Item 1A", "Item 1A.", "ITEM 1A", etc.
                re.compile(r"^\s*Item 1A\b", re.IGNORECASE),
                
                # Matches "Item 7", "Item 7.", "ITEM 7", etc.
                re.compile(r"^\s*Item 7\b", re.IGNORECASE),
                
                # Matches "Item 2", "Item 2.", "ITEM 2" (for 10-Q reports)
                re.compile(r"^\s*Item 2\b", re.IGNORECASE)
            ]
            # --------------------------------
            
            extracted_texts = []
            
            # 5. Iterate through the tree nodes (sections) to find our targets
            for node in tree.nodes:
                node_title = node.text.strip()
                
                # Check if this node's title matches ANY of our simple patterns
                for pattern in target_item_patterns:
                    if pattern.search(node_title):
                        print(f"ℹ️ [sec-parser] Found matching section: {node_title}")
                        
                        # Get all text elements within this section (node)
                        section_texts = [
                            element.text 
                            for element in node.get_descendants() 
                           
                        ]
                        
                        full_section_text = " ".join(section_texts)
                        extracted_texts.append(full_section_text)
                        
                        # Once matched, stop checking other patterns for this node
                        break 
            
            if not extracted_texts:
                print(
                    f"⚠️ [sec-parser] No Item 1A, Item 7, or Item 2 found in {filing_url}"
                )
                return None

            clean_text = " ".join(extracted_texts)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            if len(clean_text) < 1000:
                print(
                    f"⚠️ [sec-parser] Extracted text too short from {filing_url}. Skipping."
                )
                return None

            return clean_text

        except requests.exceptions.HTTPError as http_err:
            print(f"❌ [requests] HTTP Error downloading {filing_url}: {http_err}")
            return None
        except SecParserError as sec_err:
            print(f"❌ [sec-parser] Failed to parse {filing_url}: {sec_err}")
            return None
        except Exception as e:
            print(f"❌ [sec-parser] General error processing {filing_url}: {e}")
            return None
    # ==================================================================
    # SECTION 2: YAHOO FINANCE FUNCTIONS
    # ==================================================================

    def get_yfinance_news_links(self, ticker, limit=10):
        # This function is unchanged.
        print(f"ℹ️ [yfinance] Fetching news links for {ticker}...")
        try:
            ticker_obj = yf.Ticker(ticker)
            news_list = ticker_obj.news

            if not news_list:
                print(f"⚠️ [yfinance] No news links found for {ticker}.")
                return []

            links = []
            for item in news_list:
                content_data = item.get("content")
                if not content_data or not isinstance(content_data, dict):
                    continue

                title = content_data.get("title")
                click_through_url = content_data.get("clickThroughUrl", {})
                canonical_url = content_data.get("canonicalUrl", {})

                # Check both possible URL fields
                full_url = None
                if isinstance(canonical_url, dict) and canonical_url.get("url"):
                    full_url = canonical_url["url"]
                elif isinstance(click_through_url, dict) and click_through_url.get("url"):
                    full_url = click_through_url["url"]

                if not full_url or not title:
                    continue
                if not full_url.startswith("http"):
                    continue

                links.append({"url": full_url, "title": title})
                if len(links) >= limit:
                    break

            return links
        except Exception as e:
            print(f"❌ [yfinance] Failed to fetch news links for {ticker}: {e}")
            return []

    def extract_article_text(self, article_url):
        # This function is unchanged.
        try:
            article = Article(article_url, config=self.newspaper_config)
            article.download()
            article.parse()
            cleaned_text = article.text

            if len(cleaned_text) < 250:
                print(
                    f"⚠️ [newspaper3k] Text too short from {article_url} ({len(cleaned_text)} chars)"
                )
                return None

            return cleaned_text
        except Exception as e:
            print(f"❌ [newspaper3k] Failed to parse article from {article_url}: {e}")
            return None

    # ==================================================================
    # SECTION 3: COMMON FUNCTIONS
    # ==================================================================

    def clean_text(self, text, stop_phrases):
        # This function is unchanged.
        clean_text = text
        for phrase in stop_phrases:
            if phrase in clean_text:
                clean_text = clean_text.split(phrase)[0].strip()
        return clean_text

    def generate_combined_annotations(self, text, company_info, date_info):
        # --- UPDATED PROMPT: "summary" -> "key_insights" ---
        prompt = f"""
You are an expert financial analyst. Your task is to create a high-quality training dataset sample from the following financial document.
The document is regarding {company_info}, from {date_info}.

Read the text and generate:
1.  A concise 2-3 sentence "Key Insight" focusing on the most critical information (e.g., risks, strategy, or results).
2.  Two (2) factual question-answer pairs (Q&A) that a financial analyst would find insightful. The questions must be answerable *only* from the provided text, and the answers must be accurate.

Return your output as a single, valid JSON object with the keys "key_insights" and "qa_pairs".

--- FINANCIAL DOCUMENT TEXT ---
{text[:8000]}
"""

        try:
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
            response = self.model.generate_content(
                prompt, generation_config=generation_config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"❌ [Gemini] Combined-generation error: {e}")
            print("Raw output:", getattr(response, "text", "No response text"))
            return None

    # ==================================================================
    # SECTION 4: MAIN PIPELINE FUNCTIONS
    # ==================================================================

    def annotate_company(
        self, cik, form_type="10-K", limit=5, output_path="sec_dataset.json"
    ):
        # --- UPDATED with routing logic for 8-K vs 10-K ---
        print(f"--- STARTING: SEC {form_type} Workflow (CIK: {cik}) ---")

        filings = self.get_company_filings(cik, form_type=form_type, limit=limit)
        annotated_dataset = []
        if not filings:
            print(f"[SEC] No {form_type} filings found to process. Exiting.")
            return

        sec_stop_phrases = [
            "This press release contains forward-looking statements",
            "NOTE TO EDITORS:",
            "Press Contact:",
            "Investor Relations Contact:",
        ]

        for f in tqdm(filings, desc=f"[SEC] Processing {form_type} Filings"):
            text = None
            
            # --- NEW ROUTING LOGIC ---
            if form_type == "8-K":
                # Use old method for 8-K (looks for press releases)
                text = self.extract_exhibit_text(f["wrapper_url"])
            elif form_type in ("10-K", "10-Q"):
                # Use new 'sec-parser' method for 10-K/10-Q
                text = self.extract_10k_report_items(f["wrapper_url"])
            else:
                print(f"⚠️ Unsupported form type: {form_type}. Skipping.")
                continue
            # --- END ROUTING LOGIC ---

            if not text:
                print(f"ℹ️ [SEC] No usable text found for {f['wrapper_url']}. Skipping.")
                continue

            clean_text = self.clean_text(text, sec_stop_phrases)

            if len(clean_text) < 250:
                print(
                    f"⚠️ [SEC] Text for {f['wrapper_url']} too short after cleaning. Skipping."
                )
                continue

            company_info = f"{f['company']} (CIK: {f['cik']})"
            date_info = f"filing date {f['date']}"
            annotations = self.generate_combined_annotations(
                clean_text, company_info, date_info
            )

            if annotations:
                annotated_dataset.append(
                    {
                        "source": f"SEC {form_type}",
                        "company": f["company"],
                        "date": f["date"],
                        "url": f["wrapper_url"],
                        "context": clean_text,
                        "key_insights": annotations.get("key_insights"), # Updated key
                        "qa_pairs": annotations.get("qa_pairs"),
                    }
                )
            time.sleep(self.sleep_time)

        output_file = Path(output_path)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(annotated_dataset, f, indent=2, ensure_ascii=False)
        print(f"✅ [SEC] {len(annotated_dataset)} {form_type} samples saved to {output_file.name}.")

    def annotate_yahoo_news(self, ticker, limit=5, output_path="yahoo_dataset.json"):
        # This function is unchanged, except for the "summary"->"key_insights" key.
        print(f"--- STARTING: Yahoo Finance Workflow (Ticker: {ticker}) ---")
        links = self.get_yfinance_news_links(ticker, limit=limit)
        annotated_dataset = []

        if not links:
            print("[Yahoo] No news links found. Exiting.")
            return

        yahoo_stop_phrases = [
            "Read more:",
            "Related stories:",
            "Click here for the latest stock market news",
        ]

        for link in tqdm(links, desc=f"[Yahoo] Processing {ticker} News"):
            text = self.extract_article_text(link["url"])
            if not text:
                continue

            clean_text = self.clean_text(text, yahoo_stop_phrases)

            if len(clean_text) < 250:
                print(
                    f"⚠️ [Yahoo] Text for {link['url']} too short after cleaning. Skipping."
                )
                continue

            company_info = f"{ticker} (from article: {link['title']})"
            date_info = "recent news"

            annotations = self.generate_combined_annotations(
                clean_text, company_info, date_info
            )

            if annotations:
                annotated_dataset.append(
                    {
                        "source": "Yahoo/External",
                        "ticker": ticker,
                        "title": link["title"],
                        "url": link["url"],
                        "context": clean_text,
                        "key_insights": annotations.get("key_insights"), # Updated key
                        "qa_pairs": annotations.get("qa_pairs"),
                    }
                )
            time.sleep(self.sleep_time)

        output_file = Path(output_path)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(annotated_dataset, f, indent=2, ensure_ascii=False)
        print(f"✅ [Yahoo] {len(annotated_dataset)} samples saved to {output_file.name}.")


if __name__ == "__main__":
    annotator = FinancialNewsAnnotator()

    # Task 1: Generate 3 full-text examples from SEC 10-K reports for Apple (AAPL)
    # This will now use the new 'extract_10k_report_items' method
    annotator.annotate_company(
        cik="0000320193",
        form_type="10-K",  # Changed from 8-K to 10-K
        limit=3,
        output_path="apple_SEC_10K_dataset.json",
    )

    print("\n" + "=" * 50 + "\n")

    # Task 2: Generate 3 full-text examples from Yahoo Finance news for NVIDIA (NVDA)
    # annotator.annotate_yahoo_news(
    #     ticker="NVDA", limit=3, output_path="nvidia_YAHOO_dataset.json"
    # )

    # print("\n" + "=" * 50 + "\n")

    # Task 3: Generate 3 full-text examples from SEC 8-K reports for Microsoft (MSFT)
    # This will still use the original 'extract_exhibit_text' method
    # annotator.annotate_company(
    #     cik="0000789019",
    #     form_type="8-K",
    #     limit=3,
    #     output_path="msft_SEC_8K_dataset.json",
    # )