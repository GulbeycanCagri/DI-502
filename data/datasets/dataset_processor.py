import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path
from urllib.parse import urljoin
import yfinance as yf
from newspaper import Article, Config  

class FinancialNewsAnnotator:
    def __init__(self, api_key=None, model_name="gemini-flash-latest", sleep_time=1.5):
        # .env dosyasını bul
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

        # --- YENİ newspaper3k KONFİGÜRASYONU ---
        # Bu, 'newspaper' kütüphanesinin de engellenmemesi için User-Agent kullanmasını sağlar
        self.newspaper_config = Config()
        self.newspaper_config.browser_user_agent = self.headers['User-Agent']
        self.newspaper_config.request_timeout = 20
        # ----------------------------------------

    # ==================================================================
    # BÖLÜM 1: SEC EDGAR FONKSİYONLARI (Değişiklik Yok)
    # ==================================================================

    def get_company_filings(self, cik, form_type="8-K", limit=10):
        cik10 = f"{int(cik):010d}"
        cik_int = int(cik)
        json_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        filings = []
        try:
            r = requests.get(json_url, headers=self.headers, timeout=10)
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
                    if not doc: continue
                    filings.append({
                        "company": data["name"], "cik": cik10, "form": form_type, "date": date,
                        "wrapper_url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc.replace('-', '')}/{doc}"
                    })
                if len(filings) >= limit: break
            if not filings: print(f"ℹ️ [SEC] No {form_type} filings found for CIK {cik10}.")
            return filings
        except Exception as e:
            print(f"❌ [SEC] Request error for CIK {cik10}: {e}")
            return []

    def extract_exhibit_text(self, wrapper_url):
        # (Bu fonksiyon SEC için, aynı kalıyor)
        try:
            r = requests.get(wrapper_url, headers=self.headers, timeout=20)
            soup = BeautifulSoup(r.text, "html.parser")
            exhibit_link = None
            for row in soup.find_all("tr"):
                row_text = row.get_text(" ", strip=True)
                if re.search(r"EX-99\.1|Exhibit\s+99\.1", row_text, re.IGNORECASE):
                    link_tag = row.find("a", href=True)
                    if link_tag: exhibit_link = link_tag['href']; break
            if not exhibit_link:
                href_regex = re.compile(r"exh?[\-_]?99[\-_]?1", re.IGNORECASE)
                text_regex = re.compile(r"EX-99\.1|Exhibit\s+99\.1", re.IGNORECASE)
                for link_tag in soup.find_all("a", href=True):
                    if href_regex.search(link_tag.get('href', '')) or text_regex.search(link_tag.get_text(strip=True)):
                        exhibit_link = link_tag['href']; break
            if not exhibit_link:
                print(f"⚠️ [SEC] Exhibit 99.1 link not found in {wrapper_url}")
                return None
            exhibit_full_url = urljoin(wrapper_url, exhibit_link)
            ex_r = requests.get(exhibit_full_url, headers=self.headers, timeout=20)
            content_type = ex_r.headers.get('Content-Type', '').lower()
            is_html = 'html' in content_type or exhibit_full_url.endswith(('.htm', '.html'))
            cleaned = ""
            if is_html:
                ex_soup = BeautifulSoup(ex_r.text, "html.parser")
                body = ex_soup.find("body") or ex_soup
                text_blocks = [elem.get_text(" ", strip=True) for elem in body.find_all(["p", "div", "span"], recursive=True) if not elem.find("table") and elem.get_text(strip=True)]
                cleaned = " ".join(text_blocks)
            else:
                cleaned = ex_r.text
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) < 250:
                print(f"⚠️ [SEC] Extracted exhibit text too short from {exhibit_full_url}")
                return None
            return cleaned
        except Exception as e:
            print(f"⚠️ [SEC] Failed to process exhibit from {wrapper_url}: {e}")
            return None

    # ==================================================================
    # BÖLÜM 2: YAHOO FINANCE FONKSİYONLARI (GÜNCELLENDİ)
    # ==================================================================

    def get_yfinance_news_links(self, ticker, limit=10):
        """
        'yfinance' kütüphanesini kullanarak bir hisse senedi için en son haberlerin
        linklerini ve başlıklarını çeker.
        """
        print(f"ℹ️ [yfinance] Fetching news links for {ticker}...")
        try:
            ticker_obj = yf.Ticker(ticker)
            news_list = ticker_obj.news
            
            if not news_list:
                print(f"⚠️ [yfinance] No news links found for {ticker}.")
                return []

            links = []
            for item in news_list:
                content_data = item.get('content')
                if not content_data or not isinstance(content_data, dict):
                    continue
                    
                title = content_data.get('title')
                full_url = content_data.get('canonicalUrl')["url"] or content_data.get('clickThroughUrl')
                
                if not full_url or not title:
                    continue
                if not full_url.startswith('http'):
                    continue

                links.append({"url": full_url, "title": title})
                if len(links) >= limit:
                    break
            
            return links
        except Exception as e:
            print(f"❌ [yfinance] Failed to fetch news links for {ticker}: {e}")
            return []

    def extract_article_text(self, article_url):
        """
        TAM METNİ ÇEKMEK İÇİN 'newspaper3k' KULLANIR.
        Eski BeautifulSoup kodunun yerine geçer.
        """
        try:
            # newspaper3k'e URL'yi ve konfigürasyonumuzu ver
            article = Article(article_url, config=self.newspaper_config)
            
            # 1. Haberi indir
            article.download()
            
            # 2. Haberi ayrıştır (parse et)
            article.parse()
            
            # 3. Temizlenmiş tam metni al
            cleaned_text = article.text
            
            # Bazen NLP'ye ihtiyaç duyar (bu satır özet ve anahtar kelime bulmayı hızlandırır)
            # article.nlp() 
            # summary = article.summary
            
            if len(cleaned_text) < 250:
                print(f"⚠️ [newspaper3k] Text too short from {article_url} ({len(cleaned_text)} chars)")
                return None
                
            return cleaned_text
        except Exception as e:
            # Bazen bazı siteler (örn: Forbes) erişimi engeller
            print(f"❌ [newspaper3k] Failed to parse article from {article_url}: {e}")
            return None

    # ==================================================================
    # BÖLÜM 3: ORTAK KULLANILAN FONKSİYONLARI (Değişiklik Yok)
    # ==================================================================

    def clean_text(self, text, stop_phrases):
        clean_text = text
        for phrase in stop_phrases:
            if phrase in clean_text:
                clean_text = clean_text.split(phrase)[0].strip()
        return clean_text

    def generate_combined_annotations(self, text, company_info, date_info):
        prompt = f"""
You are an expert financial analyst. Your task is to create a high-quality training dataset sample from the following financial document.
The document is regarding {company_info}, from {date_info}.

Read the text and generate:
1.  A concise 2-3 sentence summary focusing on the key financial results and strategic announcements.
2.  Two (2) factual question-answer pairs (Q&A) that a financial analyst would find insightful. The questions must be answerable *only* from the provided text, and the answers must be accurate.

Return your output as a single, valid JSON object with the keys "summary" and "qa_pairs".

--- FINANCIAL DOCUMENT TEXT ---
{text[:8000]}
"""
        
        try:
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return json.loads(response.text)
        except Exception as e:
            print(f"❌ [Gemini] Combined-generation error: {e}")
            print("Raw output:", getattr(response, "text", "No response text"))
            return None

    # ==================================================================
    # BÖLÜM 4: ANA PİPELINE (İŞ AKIŞI) FONKSİYONLARI (GÜNCELLENDİ)
    # ==================================================================

    def annotate_company(self, cik, form_type="8-K", limit=5, output_path="sec_dataset.json"):
        # (Bu SEC akışında değişiklik yok)
        print(f"--- BAŞLATILIYOR: SEC 8-K İş Akışı (CIK: {cik}) ---")
        filings = self.get_company_filings(cik, form_type=form_type, limit=limit)
        annotated_dataset = []
        if not filings:
            print("[SEC] İşlenecek dosya bulunamadı. Çıkılıyor.")
            return
        sec_stop_phrases = [
            "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS",
            "CONDENSED CONSOLIDATED BALANCE SHEETS",
            "This press release contains forward-looking statements",
            "NOTE TO EDITORS:", "Press Contact:", "Investor Relations Contact:"
        ]
        for f in tqdm(filings, desc=f"[SEC] Processing {form_type} Filings"):
            text = self.extract_exhibit_text(f["wrapper_url"])
            if not text: continue
            clean_text = self.clean_text(text, sec_stop_phrases)
            if len(clean_text) < 250:
                print(f"⚠️ [SEC] Text for {f['wrapper_url']} too short after cleaning. Skipping.")
                continue
            company_info = f"{f['company']} (CIK: {f['cik']})"
            date_info = f"filing date {f['date']}"
            annotations = self.generate_combined_annotations(clean_text, company_info, date_info)
            if annotations:
                annotated_dataset.append({
                    "source": "SEC", "company": f["company"], "date": f["date"],
                    "url": f["wrapper_url"], "context": clean_text,
                    "summary": annotations.get("summary"),
                    "qa_pairs": annotations.get("qa_pairs")
                })
            time.sleep(self.sleep_time)
        output_file = Path(output_path)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(annotated_dataset, f, indent=2, ensure_ascii=False)
        print(f"✅ [SEC] {len(annotated_dataset)} adet örnek {output_file.name} dosyasına kaydedildi.")

    def annotate_yahoo_news(self, ticker, limit=5, output_path="yahoo_dataset.json"):
        """
        Yahoo News Pipeline
        (GÜNCELLENDİ: Artık 'newspaper3k' ile TAM METİN çekiyor)
        """
        print(f"--- BAŞLATILIYOR: Yahoo Finance İş Akışı (Ticker: {ticker}) ---")
        
        # 1. Sadece linkleri al
        links = self.get_yfinance_news_links(ticker, limit=limit)
        annotated_dataset = []

        if not links:
            print("[Yahoo] İşlenecek haber bulunamadı. Çıkılıyor.")
            return
            
        # Genel haber metinleri için durdurma kelimeleri
        yahoo_stop_phrases = [
            "Read more:",
            "Related stories:",
            "Click here for the latest stock market news"
        ]

        for link in tqdm(links, desc=f"[Yahoo] Processing {ticker} News"):
            
            # 2. 'newspaper3k' kullanarak TAM METNİ ÇEK
            text = self.extract_article_text(link['url'])
            if not text:
                continue # Hata mesajı fonksiyonda basıldı (örn: 403, parse hatası)

            # 3. Metni temizle
            clean_text = self.clean_text(text, yahoo_stop_phrases)

            if len(clean_text) < 250: # Artık tam metin var, limiti yükseltebiliriz
                print(f"⚠️ [Yahoo] Text for {link['url']} too short after cleaning. Skipping.")
                continue
                
            company_info = f"{ticker} (from article: {link['title']})"
            date_info = "recent news"

            # 4. Gemini ile etiketle
            annotations = self.generate_combined_annotations(clean_text, company_info, date_info)
            
            if annotations:
                annotated_dataset.append({
                    "source": "Yahoo/External", "ticker": ticker, "title": link['title'],
                    "url": link['url'], "context": clean_text, # BU ARTIK TAM METİN
                    "summary": annotations.get("summary"),
                    "qa_pairs": annotations.get("qa_pairs")
                })
            time.sleep(self.sleep_time)
            
        output_file = Path(output_path)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(annotated_dataset, f, indent=2, ensure_ascii=False)
        print(f"✅ [Yahoo] {len(annotated_dataset)} adet örnek {output_file.name} dosyasına kaydedildi.")


if __name__ =="__main__":
    
    annotator = FinancialNewsAnnotator()
    
    # # 1. Görev: Apple (AAPL) için SEC 8-K raporlarından 3 adet örnek üret
    # annotator.annotate_company(
    #     cik="0000320193",
    #     limit=3,
    #     output_path="apple_SEC_dataset.json" 
    # )
    
    print("\n" + "="*50 + "\n")

    ## 2. Görev: NVIDIA (NVDA) için harici haber kaynaklarından 3 adet TAM METİN örnek üret
    annotator.annotate_yahoo_news(
        ticker="NVDA",
        limit=3,
        output_path="nvidia_YAHOO_dataset.json"
    )
    
    print("\n"+"="*50 + "\n")
    
    # 3. Görev: Microsoft (MSFT) için SEC 8-K raporlarından 3 adet örnek üret
    # annotator.annotate_company(
    #     cik="0000789019",
    #     limit=3,
    #     output_path="msft_SEC_dataset.json"
    # )