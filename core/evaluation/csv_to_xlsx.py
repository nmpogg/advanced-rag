import pandas as pd
import numpy as np

# Đặt seed để có thể tái tạo được kết quả (tuỳ chọn)
# np.random.seed(42)

# Đọc file CSV
df = pd.read_csv('data.csv')

# Trộn chỉ 3 cột còn lại (bỏ cột qid)
# Lấy qid
qid_column = df['qid'].reset_index(drop=True)

# Trộn các hàng của 3 cột còn lại (question, cid, difficulty)
df_shuffled = df.iloc[:, 1:].sample(frac=1).reset_index(drop=True)

# Ghép lại qid với dữ liệu đã trộn
df_final = pd.concat([qid_column, df_shuffled], axis=1)

# Lưu thành file XLSX
output_file = 'data.xlsx'
df_final.to_excel(output_file, index=False, sheet_name='Questions')

print(f"✓ Đã chuyển đổi thành công!")
print(f"  File CSV: data.csv ({len(df)} dòng)")
print(f"  File XLSX: {output_file}")
print(f"\nThông tin dữ liệu:")
print(f"  - Tổng số câu hỏi: {len(df)}")
print(f"  - Số cột: {len(df.columns)}")
print(f"  - Cột: {', '.join(df.columns.tolist())}")
