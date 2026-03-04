import pandas as pd
from difflib import SequenceMatcher
import re

# Đọc file CSV
count = 0
df = pd.read_csv('data_ban_c1.csv')

print("=" * 80)
print("KIỂM TRA CÂU HỎI TRÙNG LẶP HOẶC GIỐNG NHAU")
print("=" * 80)
print(f"Tổng số câu hỏi: {len(df)}\n")

# 1. Kiểm tra trùng lặp 100%
print("\n1. CÂU HỎI TRÙNG LẶP 100%:")
print("-" * 80)
questions = df['question'].values
duplicates_found = False

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if questions[i] == questions[j]:
            print(f"  Dòng {i+2} (qid={df.iloc[i]['qid']}) và Dòng {j+2} (qid={df.iloc[j]['qid']}):")
            print(f"  Q: {questions[i]}")
            duplicates_found = True

if not duplicates_found:
    print("  ✓ Không có câu hỏi nào trùng 100%")

# 2. Kiểm tra giống nhau cao (>80%)
print("\n2. CÂU HỎI GIỐNG NHAU > 80%:")
print("-" * 80)
similar_found = False

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        q1 = questions[i].lower()
        q2 = questions[j].lower()
        
        # Tính similarity ratio
        similarity = SequenceMatcher(None, q1, q2).ratio()
        
        if similarity > 0.60 and similarity < 1.0:  # Loại bỏ trùng 100%
            print(f"  Độ giống: {similarity:.1%}")
            print(f"  Dòng {i+2} (qid={df.iloc[i]['qid']}):")
            print(f"    {questions[i]}")
            print(f"  Dòng {j+2} (qid={df.iloc[j]['qid']}):")
            print(f"    {questions[j]}")
            print()
            similar_found = True
            count += 1

if not similar_found:
    print("  ✓ Không có câu hỏi nào giống > 80%")

# 3. Thống kê độ dài câu hỏi
print("\n3. THỐNG KÊ ĐỘ DÀI CÂU HỎI:")
print("-" * 80)
lengths = df['question'].str.len()
print(f"  Có {count} câu trùng nhau:")
print(f"  Câu hỏi ngắn nhất: {lengths.min()} ký tự")
print(f"  Câu hỏi dài nhất: {lengths.max()} ký tự")
print(f"  Độ dài trung bình: {lengths.mean():.0f} ký tự")

# Tìm những câu rất ngắn
print(f"\n  Câu hỏi rất ngắn (<50 ký tự):")
short_questions = df[lengths < 50]
for idx, row in short_questions.iterrows():
    print(f"    Dòng {idx+2} (qid={row['qid']}): {row['question']}")

print("\n" + "=" * 80)
