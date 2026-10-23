# Hướng dẫn chạy, đọc và nhận xét kết quả — ISIC 2019: DenseNet121 vs CNN tự thiết kế

File này đi kèm:
- `isic2019_densenet_custom.py` — bản chạy trên server GPU
- `ISIC2019_DenseNet_Custom.ipynb` — bản Colab (cùng một code với bản `.py`)

---

## 1. Code đáp ứng yêu cầu bài tập như thế nào

| Yêu cầu (`yeucau.md`) | Notebook cũ | Bản mới |
|---|---|---|
| Dữ liệu > 5 lớp, **mỗi lớp > 500 ảnh** | **Chưa đạt**: dùng cả 8 lớp, trong đó VASC = 253 và DF = 239 ảnh | Tự động loại lớp có ≤ 500 ảnh → còn **6 lớp** (NV, MEL, BCC, BKL, AK, SCC). Biểu đồ `01_class_distribution.png` thể hiện rõ việc này |
| 1 mô hình đơn + 1 mô hình tự chế | Có (DenseNet121 + Custom_Gated_DenseResNet) | Giữ nguyên |
| Transfer learning | Có (đóng băng backbone, 5 epoch) | Có, và ghi lại điểm sau giai đoạn này (`TL_*`) |
| Fine-tuning + rã đông | Có về hình thức, **nhưng gần như không rã đông gì**: rã đông "10 tầng cuối" chỉ mở thêm 163.840/7,04 triệu tham số backbone (~2%) | Rã đông theo **khối dense** (8 khối cuối conv5 / toàn conv5 / từ giữa conv4), BatchNorm vẫn đóng băng |
| Mỗi mô hình 3 bộ tham số | Có (D1–D3, C1–C3) | Có, thiết kế lại D1–D3 để khác nhau rõ về độ sâu rã đông |

Ghi chú về cách hiểu đề: tôi hiểu "mỗi thành viên một mô hình" là mỗi người một mô hình pretrained, còn mô hình tự chế dùng chung cho nhóm. Nếu nhóm quy định khác, chỉ cần đổi `--configs`.

### Phần dư thừa đã bỏ
- 6 kiến trúc không chạy (MobileNetV2, ResNet50V2, EfficientNetB0, Custom_MS_ResCNN, Custom_CBAM_ResNet, Custom_DS_SE_DualPoolNet) cùng các lớp SE/CBAM chỉ phục vụ chúng.
- Gắn Google Drive, đường dẫn Windows `H:\...`, xử lý Unicode NFC/NFD → thay bằng `--output-dir`.
- `ModelCheckpoint` ghi file tạm mỗi epoch nhưng không bao giờ được nạp lại.
- Pickle checkpoint + chữ ký cấu hình → thay bằng mỗi run một thư mục có `result.json` (có là bỏ qua).
- Nhiều bảng trùng thông tin (stacked / compact / wide) và biểu đồ "efficiency trade-off" (số tham số, thời gian đã nằm trong bảng).
- `CONFIG` ghi đè biến toàn cục `DROPOUT_RATE` để truyền vào hàm dựng mô hình → nay truyền tham số trực tiếp.

### Vì sao notebook cũ cho điểm không cao (đã sửa)
Đây là chẩn đoán từ việc đọc code, notebook cũ chưa có kết quả chạy hoàn chỉnh để đối chiếu.
1. Rã đông quá ít (xem bảng trên) nên DenseNet gần như chỉ học lớp phân loại.
2. EarlyStopping/ReduceLR theo `val_loss` không có trọng số lớp, trong khi bài toán mất cân bằng (NV ~52%) và thước đo quan trọng là Macro-F1. Mô tả trong notebook nói chọn checkpoint theo Macro-F1 nhưng code thì không. Nay theo dõi và khôi phục trọng số theo **val Macro-F1**.
3. Mô hình tự chế train từ đầu chỉ ≤ 30 epoch, dừng sớm sau 4 epoch không cải thiện, không warm-up → dễ chưa hội tụ. Nay: ≤ 60 epoch, warm-up + cosine, patience 12.
4. Ảnh 512 bị giải mã JPEG lại ở mỗi epoch (log Colab: ~360 ms/bước với batch 32) → nay giải mã/resize 1 lần, lưu `.npy`.
5. Ảnh thu nhỏ còn 224 → mặc định 256 (đổi bằng `--img-size`).
6. Hai lớp VASC, DF rất ít ảnh làm Macro-F1 thấp → bị loại theo yêu cầu đề.

### Hai điểm sửa về phương pháp (ảnh hưởng đến con số)
- **Chia dữ liệu theo `patient_id`**: ảnh của cùng một bệnh nhân không rơi vào cả train lẫn test. Chia ngẫu nhiên thường làm điểm test cao giả. Điểm có thể thấp hơn trước, nhưng đáng tin hơn. Tắt bằng `--no-group-split`. Nếu cột `patient_id` không dùng được (gần như cùng một giá trị) code tự quay về chia phân tầng thường và in cảnh báo.
- **Grad-CAM**: bản cũ thực chất là saliency theo gradient đầu vào nhưng gọi là Grad-CAM. Bản mới là Grad-CAM thật.

---

## 2. Cách chạy

### Server GPU
```bash
pip install "tensorflow[and-cuda]>=2.16" scikit-learn pandas matplotlib seaborn openpyxl opencv-python-headless kagglehub

# 1) Chạy thử rất nhanh để kiểm tra môi trường (vài phút)
python isic2019_densenet_custom.py --data-dir /duong/dan/jpeg-isic2019-512x512 --output-dir smoke --smoke-test

# 2) Chạy thật, ghi log để không mất khi ngắt SSH
CUDA_VISIBLE_DEVICES=0 nohup python isic2019_densenet_custom.py \
    --data-dir /duong/dan/jpeg-isic2019-512x512 --output-dir outputs --img-size 256 \
    > outputs_log.txt 2>&1 &
tail -f outputs_log.txt
```
- Không có `--data-dir`: script tải bằng `kagglehub` (cần internet và tài khoản Kaggle).
- Server không có internet: tải sẵn file `densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5` rồi dùng `--backbone-weights /duong/dan/file.h5`.
- Chỉ chạy phần DenseNet của bạn: `--configs D1 D2 D3`. Run nào đã có `result.json` sẽ tự bỏ qua, ngắt giữa chừng có thể chạy lại cùng lệnh. Muốn chạy lại thì thêm `--force`.
- Hết bộ nhớ GPU: giảm batch trong `CONFIGS`. Hết RAM: giảm `--img-size`.
- Dựng lại bảng/biểu đồ mà không train: thêm `--report-only`.
- Thời gian chạy phụ thuộc GPU. Hãy đo bằng một cấu hình trước (`--configs D1`) rồi nhân lên.

### Colab
Mở notebook, sửa danh sách `ARGV` ở cell cuối, chạy lần lượt các cell. Nên đặt `--output-dir` trên Drive. Có thể chạy từng phần (`D1`, rồi `D2`...), các run đã xong được giữ lại.

### Quan trọng khi cả nhóm cùng làm
Để so sánh công bằng giữa các thành viên, **mọi người phải dùng cùng một tập chia, cùng `--img-size` và cùng `--seeds`**. Một người chạy trước, rồi gửi file `outputs/split.csv` cho các bạn khác chạy với `--split-file split.csv`.

---

## 3. Các file kết quả

```
outputs/
├── split.csv                       # tập nào (train/val/test) của từng ảnh — dùng chung cả nhóm
├── reports/
│   ├── summary.md                  # bảng sẵn để dán vào báo cáo
│   ├── summary_by_config.csv       # số liệu đầy đủ (mean, std) theo cấu hình
│   ├── transfer_vs_finetune.csv    # DenseNet: chỉ đóng băng vs đóng băng + rã đông
│   ├── per_class_f1.csv, per_class_recall.csv
│   ├── runs_raw.csv                # mỗi dòng là 1 lần chạy
│   ├── results.xlsx                # gộp các bảng trên (mỗi bảng một sheet)
│   ├── dataset_split_summary.csv   # số ảnh mỗi lớp ở train/val/test
│   └── run_config.json             # tham số đã dùng (để ghi vào phần phương pháp)
├── figures/                        # 01 … 09 (xem mục 4)
└── runs/<Cfg>_seed<N>/             # model.keras, history.json, test_probs.npy, result.json
```

---

## 4. Cách đọc và nhận xét từng kết quả

### 4.1 Nguyên tắc chung
- **Chọn cấu hình tốt nhất bằng Validation** (cột `Val Macro-F1`, cột `Best_By_Val` trong `summary_by_config.csv`). **Báo cáo số liệu Test**. Không chọn cấu hình theo điểm Test, vì như vậy là dùng đáp án để chọn mô hình.
- Dữ liệu mất cân bằng (NV chiếm hơn một nửa). Một mô hình đoán toàn NV vẫn đạt Accuracy ~52%. Vì vậy **hãy nhìn Macro-F1 và Balanced Accuracy trước, Accuracy sau**.
- Chỉ chạy 1 seed thì không có độ lệch chuẩn (`±`). Khi hai cấu hình chênh nhau dưới ~1 điểm phần trăm, đừng khẳng định cái nào tốt hơn. Muốn kết luận chắc, chạy `--seeds 42 123 999` (gấp 3 thời gian).

### 4.2 Bảng chỉ số (`summary.md`)
| Chỉ số | Ý nghĩa | Cách nhận xét |
|---|---|---|
| Accuracy | Tỷ lệ đoán đúng toàn bộ | Dễ bị lớp NV kéo cao, chỉ nhận xét kèm các chỉ số dưới |
| Balanced Acc = Macro Recall | Trung bình recall từng lớp | Mỗi lớp quan trọng như nhau; chỉ số chính để so sánh |
| Macro Precision | Khi mô hình báo một lớp, đúng bao nhiêu | Thấp = hay báo nhầm (nhiều dương tính giả) |
| Macro Recall | Bỏ sót bao nhiêu | Thấp = hay bỏ sót, nghiêm trọng với MEL, BCC, SCC |
| Macro-F1 | Điều hòa Precision và Recall | **Chỉ số tổng hợp chính** |
| Weighted F1 | Như F1 nhưng theo số mẫu từng lớp | Gần Accuracy, nghiêng về NV |
| Macro ROC-AUC | Khả năng xếp hạng đúng, không phụ thuộc ngưỡng | Cao mà F1 thấp = mô hình "biết" nhưng ngưỡng quyết định chưa tốt |
| Train (phút), Params (M), Train params (M) | Chi phí | So sánh độ phức tạp: DenseNet121 ~7,3M tham số (chỉ một phần được train), mô hình tự chế ~2,5M (train toàn bộ) |

Khung nhận xét mẫu (điền số của bạn):
> Trong 3 cấu hình DenseNet121, **D_ có Val Macro-F1 cao nhất** (0.___), trên Test đạt Accuracy ___, Balanced Accuracy ___ và Macro-F1 ___. So với mô hình tự thiết kế tốt nhất (C_: Macro-F1 ___), DenseNet121 cao/thấp hơn ___ điểm phần trăm, trong khi dùng ___ lần số tham số và thời gian huấn luyện ___ phút so với ___ phút.

### 4.3 Transfer learning vs Fine-tuning (`transfer_vs_finetune.csv`)
Cột `TL` là điểm sau giai đoạn đóng băng backbone, `TL+FT` là điểm cuối cùng sau rã đông, `Δ` là mức chênh. Đây là bằng chứng cho yêu cầu "có fine-tuning, transfer learning, rã đông".
- Δ dương rõ rệt: fine-tuning có ích, đặc trưng ImageNet cần điều chỉnh cho ảnh dermoscopy (khác ảnh tự nhiên).
- Δ ≈ 0 hoặc âm: rã đông không giúp trong cấu hình đó (LR quá cao, hoặc quá ít dữ liệu cho phần được rã đông); nêu đúng như vậy, đừng bỏ qua.
- So D1 → D2 → D3: rã đông càng sâu có cho điểm cao hơn không? Thường phải đánh đổi: rã đông sâu thì nhiều tham số được huấn luyện hơn, thời gian lâu hơn, dễ overfit hơn.

### 4.4 Learning curves (`04_learning_curves.png`)
Mỗi hàng là một cấu hình; 3 cột: Loss, Accuracy, Val Macro-F1. Đường tím chấm là chỗ bắt đầu fine-tuning (chỉ có ở DenseNet).
- **Overfitting:** train loss tiếp tục giảm trong khi val loss tăng lên hoặc đi ngang → khoảng cách ngày càng lớn. Nhận xét kèm dropout và việc dừng sớm.
- **Underfitting:** cả train và val đều cao/thấp và vẫn đang cải thiện khi hết epoch → cần train lâu hơn (mô hình tự chế hay gặp).
- **Không phải lỗi:** train accuracy thấp hơn val ở epoch đầu là bình thường, vì train có augmentation và dropout còn val thì không.
- Sau đường tím, val Macro-F1 nhảy lên = fine-tuning có hiệu quả. Val Macro-F1 dao động mạnh = LR hơi cao.
- Trọng số được khôi phục về **epoch có val Macro-F1 cao nhất** (có dòng `[BEST]` trong log), nên điểm Test lấy từ epoch đó, không phải epoch cuối của đường cong.

### 4.5 Confusion matrix (`05_confusion_matrices.png`)
Chuẩn hóa theo hàng (%): hàng = lớp thực, cột = lớp dự đoán, **đường chéo = recall của lớp**.
- Tìm các ô ngoài đường chéo có giá trị cao = cặp lớp bị nhầm. Cặp hay gặp trong ISIC: **MEL ↔ NV** (cùng là tổn thương sắc tố), **BKL ↔ NV/MEL**, **AK ↔ SCC ↔ BKL** (đều là dày sừng/tổn thương vảy).
- Nêu rõ **recall của MEL**: bỏ sót ung thư hắc tố nguy hiểm hơn báo nhầm. Một mô hình Accuracy cao nhưng recall MEL thấp thì kém hơn về ý nghĩa y khoa.
- Lớp ít mẫu (AK, SCC) thường có đường chéo thấp nhất; giải thích bằng số ảnh train (`dataset_split_summary.csv`) và trọng số lớp.

### 4.6 ROC (`06_roc_curves.png`)
Đường macro-average (trung bình các lớp, one-vs-rest). Càng gần góc trên bên trái càng tốt; AUC ghi trong chú giải. AUC giữa các cấu hình thường sát nhau (vì dễ đạt cao), nên đừng dùng AUC một mình để chọn mô hình, hãy dựa vào Macro-F1.

### 4.7 So sánh chỉ số và F1 theo lớp (`07`, `08`)
- `07`: 5 chỉ số cạnh nhau cho mọi cấu hình; thanh sai số chỉ có khi chạy nhiều seed.
- `08`: bản đồ nhiệt F1 theo lớp. Xem lớp nào mô hình yếu (ô đỏ). Mô hình tốt về trung bình nhưng 0 ở một lớp cần được nêu rõ.
- Nhận xét nên so sánh **cùng lớp** giữa hai mô hình: mô hình pretrained thường hơn rõ ở các lớp ít mẫu vì tận dụng đặc trưng học từ ImageNet; mô hình tự chế phải học từ đầu nên khó hơn với lớp ít dữ liệu.

### 4.8 Grad-CAM (`09_gradcam_<mô hình>.png`)
Cho cấu hình tốt nhất (theo Val Macro-F1) của từng mô hình. Màu đỏ/vàng = vùng ảnh ảnh hưởng nhiều nhất đến quyết định. Tiêu đề xanh = đoán đúng, đỏ = đoán sai.
- Tốt: nhiệt tập trung trên tổn thương.
- Đáng ngờ: nhiệt rơi vào nền da, lông, thước đo, viền tối góc ảnh → mô hình có thể học "lối tắt" (đặc điểm không liên quan bệnh). Nêu hạn chế này nếu thấy.
- Mỗi lớp chỉ hiện 1 ảnh (ảnh test đầu tiên của lớp), vì vậy chỉ minh họa, không đủ để kết luận về cả mô hình.

---

## 5. Gợi ý cấu trúc phần "Kết quả và thảo luận" trong báo cáo
1. **Dữ liệu:** `01_class_distribution.png`, nêu lý do loại VASC và DF (≤ 500 ảnh), bảng `dataset_split_summary.csv`, cách chia 70/15/15 theo bệnh nhân.
2. **Phương pháp:** hai mô hình, 2 giai đoạn của DenseNet, bảng 3 bộ tham số (D1–D3, C1–C3), AdamW, label smoothing, class weight (`run_config.json`).
3. **Kết quả:** bảng `summary.md`, `04`, `05`, `06`, `07`, `08`.
4. **Transfer vs fine-tuning:** `transfer_vs_finetune.csv`.
5. **Giải thích mô hình:** Grad-CAM, phân tích các cặp lớp nhầm lẫn.
6. **Hạn chế:** 1 seed (nếu vậy), tập test của một bộ dữ liệu, không phải công cụ chẩn đoán lâm sàng, chưa kiểm chứng trên bộ dữ liệu ngoài.

## 6. Lưu ý trung thực
- Script đã được chạy thử thành công trên dữ liệu giả (CPU, cả đường chạy `mixed_float16`/`mixed_bfloat16`, resume, `--report-only`, Grad-CAM, notebook) nhưng **chưa chạy trên ISIC 2019 thật và GPU thật**. Chưa có con số kết quả thật nào; tôi không thể dự đoán điểm bạn sẽ đạt.
- Bộ tham số D1–D3, C1–C3 là điểm xuất phát hợp lý, chưa được tinh chỉnh. Nếu một cấu hình rõ ràng kém, hãy sửa `CONFIGS` trong code và ghi lại trong báo cáo.
- Số lớp, số ảnh và cột `patient_id` trong dữ liệu thật được script in ra ở đầu log (`[CLASSES]`, `[SPLIT]`); hãy kiểm tra các dòng này khi chạy lần đầu.
