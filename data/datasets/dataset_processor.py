import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import google.generativeai as genai
import requests
import sec_parser as sp
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from newspaper import Article, Config
from sec_downloader import Downloader
from sec_parser import TreeBuilder
from sec_parser.exceptions import SecParserError
from tqdm import tqdm


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
        self.sec_headers = {"User-Agent": "Economind Project economind@example.com"}

        self.newspaper_config = Config()
        self.newspaper_config.browser_user_agent = self.headers["User-Agent"]
        self.newspaper_config.request_timeout = 20

        self.sec_downloader = Downloader("Economind Team", "e274086@metu.edu.tr")

    def get_company_filings(
        self, cik, form_type="8-K", limit=10, before_date=None, after_date=None
    ):
        """
        Fetch SEC filings for a given CIK with optional date filters.

        Args:
            cik (str or int): Company CIK number.
            form_type (str): SEC form type, e.g., "10-K", "8-K", "10-Q".
            limit (int): Max number of filings to return.
            before_date (str): Optional filter. Include only filings before this date (format: 'YYYY-MM-DD').
            after_date (str): Optional filter. Include only filings after this date (format: 'YYYY-MM-DD').

        Returns:
            list[dict]: Filtered filings.
        """
        cik10 = f"{int(cik):010d}"
        cik_int = int(cik)
        json_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        filings = []

        # Convert date strings to datetime objects for comparison
        before_dt = datetime.strptime(before_date, "%Y-%m-%d") if before_date else None
        after_dt = datetime.strptime(after_date, "%Y-%m-%d") if after_date else None

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
                if form != form_type:
                    continue

                acc, doc, date_str = accession_nums[i], primary_docs[i], filing_dates[i]
                if not (acc and doc and date_str):
                    continue

                # Parse filing date for filtering
                try:
                    filing_dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue

                # Apply optional date filters
                if before_dt and filing_dt >= before_dt:
                    continue
                if after_dt and filing_dt <= after_dt:
                    continue

                filings.append(
                    {
                        "company": data["name"],
                        "cik": cik10,
                        "form": form_type,
                        "date": date_str,
                        "wrapper_url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc.replace('-', '')}/{doc}",
                    }
                )

                if len(filings) >= limit:
                    break

            if not filings:
                print(
                    f"ℹ️ [SEC] No {form_type} filings found for CIK {cik10} with given filters."
                )
            return filings

        except Exception as e:
            print(f"❌ [SEC] Request error for CIK {cik10}: {e}")
            return []

    def extract_exhibit_text(self, wrapper_url):
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

    def extract_10k_report_items(self, filing_url):
        """
        Downloads the HTML from the given filing_url using 'requests'
        and parses it to find and extract the text of specific items.
        This version uses simple, robust regex patterns.
        """
        try:
            print(f"ℹ️ [sec-parser] Downloading and Parsing: {filing_url}")

            r = requests.get(filing_url, headers=self.sec_headers, timeout=20)
            r.raise_for_status()
            filing_html = r.text

            if not filing_html:
                print(f"⚠️ [requests] Downloaded HTML from {filing_url} is empty.")
                return None

            parser = sp.Edgar10QParser()
            elements = parser.parse(filing_html)

            tree = TreeBuilder().build(elements)

            target_item_patterns = [
                # Most specific patterns first
                # Matches "Item 1A", "Item 1A.", etc. (Risk Factors)
                re.compile(r"^\s*Item 1A\b", re.IGNORECASE),
                # Matches "Item 7A", "Item 7A.", etc. (Market Risk)
                re.compile(r"^\s*Item 7A\b", re.IGNORECASE),
                # Less specific patterns last
                # Matches "Item 1", "Item 1.", etc. (Business Overview)
                # This comes *after* 1A so it doesn't match "Item 1A"
                re.compile(r"^\s*Item 1\b", re.IGNORECASE),
                # Matches "Item 2", "Item 2.", etc. (MD&A for 10-Q)
                re.compile(r"^\s*Item 2\b", re.IGNORECASE),
                # Matches "Item 3", "Item 3.", etc. (Legal Proceedings)
                re.compile(r"^\s*Item 3\b", re.IGNORECASE),
                # Matches "Item 7", "Item 7.", etc. (MD&A for 10-K)
                # This comes *after* 7A
                re.compile(r"^\s*Item 7\b", re.IGNORECASE),
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
                            element.text for element in node.get_descendants()
                        ]

                        full_section_text = " ".join(section_texts)
                        extracted_texts.append(full_section_text)

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

    def get_yfinance_news_links(self, ticker, limit=10):
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
                elif isinstance(click_through_url, dict) and click_through_url.get(
                    "url"
                ):
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

    def clean_text(self, text, stop_phrases):
        clean_text = text
        for phrase in stop_phrases:
            if phrase in clean_text:
                clean_text = clean_text.split(phrase)[0].strip()
        return clean_text

    def generate_combined_annotations(self, text, company_info, date_info):
        prompt = f"""
You are an expert financial analyst. Your task is to create a high-quality training dataset sample requiring deep analytical inference from the following financial document.
The document is regarding {company_info}, from {date_info}.

Read the text and generate:
1. A concise 2-3 sentence "Key Insight" focusing on the most critical information (e.g., risks, strategy, or results).
2. Two (2) factual question-answer pairs (Q&A) that require analytical depth and synthesis.

### Instructions to Increase Question Difficulty (Requiring Analytical Inference)

It is mandatory that your questions avoid being easily answerable by simple RAG (Direct Retrieval) mechanisms. The questions must:

a) **Require Multi-Step Synthesis:** To answer the question, the model must synthesize or combine **at least two separate pieces of information (A and B)** from different sentences or sections of the text to arrive at the final answer (C).
b) **Mandatory Inference/Calculation:** The question must target an outcome that is **not explicitly stated** but is definitively calculable or logically inferable based on the data presented in the text. (e.g., combining 'cost from Section A' and 'revenue loss from Section B' to determine the 'Total Net Impact').
c) **Constraint Reinforcement:** The answers must be accurately and definitively verifiable *only* using the information provided within the text. The answers must be factual.

3. The two Q&A pairs must cover distinct aspects of the document (e.g., one risk, one financial outcome) and must strictly avoid overly generic, single-sentence lookup questions.

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

    def annotate_company(
        self,
        cik,
        form_type="10-K",
        before_date: Optional[str] = None,
        after_date: Optional[str] = None,
        limit=5,
        output_path="sec_dataset.json",
    ):
        print(f"--- STARTING: SEC {form_type} Workflow (CIK: {cik}) ---")

        filings = self.get_company_filings(
            cik,
            form_type=form_type,
            before_date=before_date,
            after_date=after_date,
            limit=limit,
        )
        for f in filings:
            print(f["wrapper_url"])

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

            if form_type == "8-K":
                text = self.extract_exhibit_text(f["wrapper_url"])
            elif form_type in ("10-K", "10-Q"):
                text = self.extract_10k_report_items(f["wrapper_url"])
            else:
                print(f"⚠️ Unsupported form type: {form_type}. Skipping.")
                continue

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
                        "key_insights": annotations.get("key_insights"),  # Updated key
                        "qa_pairs": annotations.get("qa_pairs"),
                    }
                )
            time.sleep(self.sleep_time)

        output_file = Path(output_path)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(annotated_dataset, f, indent=2, ensure_ascii=False)
        print(
            f"✅ [SEC] {len(annotated_dataset)} {form_type} samples saved to {output_file.name}."
        )

    def annotate_yahoo_news(self, ticker, limit=5, output_path="yahoo_dataset.json"):
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
                        "key_insights": annotations.get("key_insights"),  # Updated key
                        "qa_pairs": annotations.get("qa_pairs"),
                    }
                )
            time.sleep(self.sleep_time)

        output_file = Path(output_path)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(annotated_dataset, f, indent=2, ensure_ascii=False)
        print(
            f"✅ [Yahoo] {len(annotated_dataset)} samples saved to {output_file.name}."
        )


if __name__ == "__main__":
    annotator = FinancialNewsAnnotator()

    # TODO: Adjust CIKs and parameters as needed
    companies = {
        "Apple": "0000320193",
        "Microsoft": "0000789019",
        "NVIDIA": "0001045810",
        "Tesla": "0001318605",
        "Amazon": "0001018724",
        "Alphabet": "0001652044",
        "Meta": "0001326801",
        "Intel": "0000050863",
        "Netflix": "0001065280",
        "Adobe": "0000796343",
        "Walmart": "0000104169",
        "JPMorgan Chase & Co.": "0000019617",
        "Visa Inc.": "0001403161",
        "Broadcom Inc.": "0001715729",
        "Exxon Mobil Corporation": "0000034088",
        "Salesforce": "0001108524",
        "PayPal Holdings, Inc.": "0001633917",
        "Starbucks Corporation": "0000877782",
        "The Walt Disney Company": "0001001039",
        "Boeing Company": "0000012927",
    }

    test_dataset_path = "test_SEC_10K_dataset"
    os.makedirs(test_dataset_path, exist_ok=True)
    for name, cik in companies.items():
        annotator.annotate_company(
            cik=cik,
            form_type="10-K",
            limit=15,
            before_date="2025-01-01",
            output_path=os.path.join(
                test_dataset_path, f"{name.lower()}_SEC_10K_dataset.json"
            ),
        )

    print("\n" + "=" * 50 + "\n")

    # annotator.annotate_yahoo_news(
    #     ticker="NVDA", limit=10, output_path="nvidia_YAHOO_dataset.json"
    # )

    # print("\n" + "=" * 50 + "\n")
