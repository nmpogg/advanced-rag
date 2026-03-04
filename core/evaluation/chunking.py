import fitz  # PyMuPDF
import re
import json
import pandas as pd

class LawPDFProcessor:
    def __init__(self, links):
        self.links = links
        self.full_text = ""
        self.chunks = []
        self.count = 0

    def clean_text(self, text):
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_text_from_pdf(self):
        doc = fitz.open(self.links)
        raw_text = []

        for page in doc:
            text = page.get_text()
            lines = text.split('\n')

            clean_lines = []
            for line in lines:
                # print(f"Debug line: {line}")
                # bỏ qua quốc hiệu, tiêu ngữ mở đầu
                if any(x in line for x in ["Luật số:", "QUỐC HỘI", "Độc lập - Tự do", "CỘNG HÒA"]):
                    continue
                clean_lines.append(line)
            
            raw_text.append("\n".join(clean_lines))
        self.full_text = "\n".join(raw_text)

    def parse_structure(self):
        """Tách Chương -> Điều -> Khoản"""
        
        pattern = r"(?:\n|^)(Điều \d+\..*?)(?=\nĐiều \d+\.|(?:\nChương [IVX]+)|$)"
        
        matches = re.finditer(pattern, self.full_text, re.DOTALL)
        
        count = 0
        for match in matches:
            count += 1
            art_raw = match.group(1).strip()
            
            # Tách Điều thành 2 phần: [Tiêu đề + Dẫn nhập] và [Các khoản 1, 2, 3...]  
            # Regex tìm "1." đứng đầu dòng hoặc sau tiêu đề
            first_clause_match = re.search(r'(?:\n|^)\s*1\.\s', art_raw)
            
            if first_clause_match:
                # Phần trước "1." chính là Tiêu đề đầy đủ
                title_part = art_raw[:first_clause_match.start()].strip()
                # Phần sau là nội dung các khoản
                body_part = art_raw[first_clause_match.start():]
                
                # Clean title để lưu vào metadata đẹp hơn
                art_title = self.clean_text(title_part)

                # Tách các khoản: Tìm \n + số + dấu chấm + khoảng trắng
                clauses = re.split(r'(?:\n|^)(?=\d+\.\s)', body_part)
                clauses = [c for c in clauses if c.strip()]
                
                for clause in clauses:
                    clause_clean = self.clean_text(clause)
                    clause_num_match = re.match(r'(\d+)\.', clause.strip())
                    clause_num = clause_num_match.group(1) if clause_num_match else ""
                    
                    # Luật > Tên Điều > Nội dung khoản
                    enriched_content = f"{art_title}. Khoản {clause_clean}"
                    
                    self.add_chunk(enriched_content, art_title, "clause", clause_num)

            else:
                
                # Tách tiêu đề ở dấu chấm đầu tiên hoặc xuống dòng đầu tiên
                # lấy dòng đầu tiên làm title metadata
                split_idx = art_raw.find('\n')
                if split_idx != -1:
                    title_part = art_raw[:split_idx]
                else:
                    # Nếu ngắn quá không có xuống dòng, lấy hết làm title
                    title_part = art_raw.split('.')[0] + "."
                
                art_title = self.clean_text(title_part)
                content_clean = self.clean_text(art_raw)
                
                enriched_content = f"{art_title}: {content_clean}"
                self.add_chunk(enriched_content, art_title, "article")

        print(f"Đã xử lý {count} điều luật.")

    def add_chunk(self, content, article_title, type, clause_num=""):
        """Tạo bản ghi JSON chuẩn"""
        self.count += 1
        chunk_id = self.count
        
        try:
            art_num = re.search(r"Điều (\d+)", article_title).group(1)
        except:
            art_num = "0"

        record = {
            "cid": chunk_id,
            "content": content,
            "metadata": {
                "article": article_title,
                "article_num": art_num,
                "clause_num": clause_num,
                "type": type,
                "length": len(content)
            }
        }
        self.chunks.append(record)

    def save_to_json(self, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"lưu {len(self.chunks)} bản ghi vào file: {output_path}")

    def json_to_csv(self, json_path, csv_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        df = df[['cid', 'content']]
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Đã chuyển {len(data)} bản ghi từ {json_path} sang {csv_path}")


if __name__ == "__main__":
    pdf_path = "Luật-35-2024-QH15.pdf" 

    processor = LawPDFProcessor(pdf_path)
    
    processor.extract_text_from_pdf()
    
    processor.parse_structure()
    
    processor.save_to_json("corpus.json")
    
    processor.json_to_csv("corpus.json", "corpus.csv")