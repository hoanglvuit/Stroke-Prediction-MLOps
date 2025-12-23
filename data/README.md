## Giới thiệu bộ dữ liệu dự đoán đột quỵ

Bộ dữ liệu `healthcare-dataset-stroke-data.csv` chứa thông tin nhân khẩu học và lâm sàng của bệnh nhân, được sử dụng cho bài toán **dự đoán nguy cơ đột quỵ**. Mỗi dòng tương ứng với một bệnh nhân, mỗi cột là một thuộc tính (feature).

### Các thuộc tính trong dữ liệu

- **id**: Mã định danh duy nhất cho mỗi bệnh nhân.
- **gender**: Giới tính của bệnh nhân, có thể là `"Male"`, `"Female"` hoặc `"Other"`.
- **age**: Tuổi của bệnh nhân (đơn vị: năm).
- **hypertension**: Tình trạng tăng huyết áp:
  - `0`: Không bị tăng huyết áp  
  - `1`: Có chẩn đoán tăng huyết áp
- **heart_disease**: Tình trạng bệnh tim:
  - `0`: Không mắc bệnh tim  
  - `1`: Có mắc bệnh tim
- **ever_married**: Tình trạng hôn nhân, `"No"` (chưa từng kết hôn) hoặc `"Yes"` (đã từng/kết hôn).
- **work_type**: Loại hình công việc:
  - `"children"`: Trẻ em (chưa đi làm)  
  - `"Govt_job"`: Nhân viên nhà nước  
  - `"Never_worked"`: Chưa từng đi làm  
  - `"Private"`: Làm việc trong khu vực tư nhân  
  - `"Self-employed"`: Tự kinh doanh
- **Residence_type**: Loại nơi cư trú, `"Rural"` (nông thôn) hoặc `"Urban"` (thành thị).
- **avg_glucose_level**: Mức đường huyết trung bình trong máu (đơn vị đo theo bộ dữ liệu gốc).
- **bmi**: Chỉ số khối cơ thể (Body Mass Index - BMI).
- **smoking_status**: Tình trạng hút thuốc:
  - `"formerly smoked"`: Đã từng hút thuốc nhưng đã bỏ  
  - `"never smoked"`: Chưa từng hút thuốc  
  - `"smokes"`: Đang hút thuốc  
  - `"Unknown"`: Không có thông tin về tình trạng hút thuốc
- **stroke**: Nhãn mục tiêu cho biết bệnh nhân có từng bị đột quỵ hay không:
  - `1`: Bệnh nhân đã từng bị đột quỵ  
  - `0`: Bệnh nhân chưa từng bị đột quỵ

Bộ dữ liệu này thường được dùng cho các bài toán **phân loại nhị phân** (có/không đột quỵ), khai phá dữ liệu y tế và xây dựng mô hình **Machine Learning / MLOps** trong lĩnh vực chăm sóc sức khỏe.

