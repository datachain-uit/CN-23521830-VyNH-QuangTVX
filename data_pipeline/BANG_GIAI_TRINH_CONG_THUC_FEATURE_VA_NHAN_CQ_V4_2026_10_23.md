# Bảng giải trình công thức feature và nhãn CQ V4

Ngày đối chiếu: **23/10/2026**. Phạm vi: `input` → dim/bridge → event stages → `final_quality_v4` và `phase_features_v4`. Đối chiếu mã nguồn, notebook, SQL checkpoint, manifest và cấu hình đang có; tái kiểm preprocessing và vocabulary bằng **train/dev**. Không chạy lại builder/sentiment và không đọc feature/nhãn từng dòng test.

Tài liệu này giải trình **cách thực sự đang tính**. Đây là hồ sơ để xem xét O24, chưa phải văn bản cô phê duyệt công thức, đánh giá độ chính xác sentiment hay biên bản kiểm tra thủ công D06. Tài liệu nằm ngoài các gói dữ liệu đã có checksum.

## 1. Kết quả đối chiếu chính

- Learner có **62 feature: 57 numeric + 5 categorical**. Selector dùng **56 numeric**, bỏ `phase_fraction` và cả 5 categorical. `COELO/AFELO/ACELO/q/y` là thành phần nhãn/target, không thuộc X.
- Công thức nhãn V4 kế thừa bản `final-quality-v2-p95-signed-fa-2026-10-20`. V3 lọc cặp chỉ có chương `-1`; V4 đổi split sang họ khóa, giữ nguyên các giá trị nhãn đã lưu.
- **P95 của nhãn** là reference cuối cửa sổ của quần thể V2 đã đóng băng. **Scaler X V4** là median + mean/std fit trên 527.624 snapshot train V4. Hai phép này có phạm vi và mục đích khác nhau.
- Hash hiện tại của `15_build_final_quality_v2.py` khác hash trong manifest lần chạy. Tuy nhiên, **5 bước SQL trọng yếu** lưu ở checkpoint khớp chính xác với code hiện tại. Giữ cả hai hash trong mục 11; chưa khẳng định toàn bộ script là đúng bản lịch sử.
- Kiểm tra mới: **57/57** bộ tham số numeric khớp; **8/8** bảng raw→ready train/dev có 0 sai lệch; **4/4** vocabulary đúng tập token train. Bằng chứng: [evidence.json](formula_feature_v4_audit/evidence.json).

## 2. Ký hiệu và cửa sổ thời gian dùng chung

| Ký hiệu | Định nghĩa triển khai |
|---|---|
| `u,c,i,v` | Người học, course, big chapter/LO và video. Chapter là thành phần số đầu của chuỗi chapter, không phải từng video. |
| `e` | `date_trunc('day', enroll_time_raw)`, tức ngày ghi danh lúc 00:00:00 theo quy ước timestamp của pipeline. |
| `T` | Ngày có hoạt động cuối của course, làm tròn thành 23:59:59; được suy ra hồi cứu từ event, không phải lịch kết thúc hành chính. |
| `t_p` | `epoch_us(e) + ((epoch_us(T)-epoch_us(e))*a)//b`, đổi lại thành timestamp; chia nguyên ở microsecond. P1: 1/4, P2: 1/2, P3: 3/4, P4: 9/10. |
| `W_p` | Event đúng người/course, có `e ≤ event_time ≤ t_p`; hai đầu bao gồm. FullTrain-A dùng cutoff `T`. |
| `W_final` | Cửa sổ tạo target cuối kỳ: `e ≤ event_time ≤ T`. Target giữ nguyên qua bốn phase. |
| `mean_available` | Trung bình các giá trị khác NULL; 0 hợp lệ vẫn được tính. Không có thành phần khả dụng → NULL. |
| `clip(x,l,h)` | `min(h,max(l,x))`. Chỉ dùng tại các công thức ghi rõ clip. |
| `P95+(x)` | `quantile_cont(x,0.95) FILTER(WHERE x>0)`, tính trong đúng nhóm reference. |
| `P95abs(FA_raw)` | `quantile_cont(abs(FA_raw),0.95) FILTER(WHERE abs(FA_raw)>0)`. |
| `D` | Một ngày = 86.400 giây. Đếm ngày hoạt động bằng ngày UTC của thời điểm bắt đầu event. |

Điều kiện thời gian hợp lệ gồm khóa tồn tại trong catalog, timestamp đầu/cuối hợp lệ và `e<T`. Không yêu cầu người học đã có hoạt động ở phase sớm. Video epoch giây; comment epoch millisecond chia 1.000; problem/reply parse timestamp chuỗi. Code đặt timezone UTC và không có bước hiệu chỉnh múi giờ nguồn. [F05], [F06], [F15], [F16].

## 3. Dòng dữ liệu từ input và các bảng dim/bridge

### 3.1. Input và chuyển sang Parquet

`01_moocubex_raw_to_parquet.ipynb`, cell code index 3, khai báo schema cho các bảng lớn; cell 6 chuyển JSON/TXT/TSV sang Parquet. `input/moocubex_parquet/manifest.json` lưu 31 nguồn. Đây là lớp chuyển định dạng; chưa tạo chỉ số chất lượng.

| Nguồn thực tế trong `input/moocubex_parquet/tables/` | Trường dùng | Đi tiếp tới đâu |
|---|---|---|
| `entities/course` | `id,resource,field,name` | Catalog course, chapter, video/exercise; field signature. |
| `entities/problem` + `relations/exercise_problem` | `problem_id,exercise_id,score` và cặp ánh xạ | Max score, problem trong chapter, mẫu số academic/coverage. |
| `entities/video` + `relations/video_id_ccid` | `ccid,end,video_id` | Duration proxy trong catalog; tập nội dung ccid để tạo course family V4. |
| `entities/user` | `id,course_order,enroll_time,year_of_birth,gender` | Enrollment, tuổi, gender và learner vocabulary. |
| `relations/user_video` | `user_id,seq[].video_id,segment[].{local_start_time,start_point,end_point,speed}` | Video event và các khoảng nội dung đã xem. |
| `relations/user_problem` | `log_id,user_id,problem_id,is_correct,attempts,score,submit_time` | Dòng trạng thái bài tập tại submit_time; không phải timestamp của từng attempt. |
| `entities/translated_comment` + `relations/course_comment` | `id,user_id,resource_id,text,create_time` và ánh xạ course | Comment, sentiment, chương biết/không biết. |
| `entities/reply` + `relations/comment_reply` | `id,user_id,text,create_time`, cặp comment–reply | Reply tác giả/nhận, kế thừa resource và chapter từ comment cha. |
| `relations/course_teacher`, `relations/course_school` | Tập teacher/school của course | Signature và encoder categorical. |

Pipeline dùng **translated_comment** cho sentiment nhưng dùng **reply gốc**, không suy luận sentiment riêng cho reply. `translated_reply` có trong input nhưng không phải nguồn inference hiện hành. Chưa thấy manifest mô tả model/phiên bản dịch đã tạo `translated_comment`; không gán phiên bản dịch giả định.

### 3.2. Các bảng dim/bridge thực sự được tạo

Notebook [F02], cell index 8/9/11/13/15, tạo các bảng dưới `input/moocubex_model/processed/`. Số dòng sau đây lấy từ `build_manifest.json`, không phải đếm lại trong lần giải trình này.

| Bảng | Số dòng đã ghi | Grain/quy tắc và vai trò |
|---|---:|---|
| `dim_course` | 3.781 | Một course; chuẩn ID, chọn bản nhiều resource hơn nếu trùng ID. Giữ field/context. |
| `dim_problem` | 2.454.422 | Một problem; `problem_max_score=max(score≥0)`; ghi trạng thái thiếu/âm hoặc xung đột score. |
| `dim_video` | 59.581 | Một ccid; duration proxy từ max caption end dương. Proxy này **không phải mẫu số TS V4 hiện hành**. |
| `dim_lo_resource` | 271.043 | Một course–big chapter–resource–type; `lo_key=course_id:CH{chapter_major}`; deduplicate, đánh dấu resource ánh xạ nhiều nơi. |
| `dim_resource` | 271.043 | Tổng hợp một resource; thống kê số course/chapter/type phục vụ audit ánh xạ. |
| `bridge_video_ccid` | 2.798.892 | Cặp video_id–ccid duy nhất và độ nhập nhằng. V4 family đọc quan hệ raw ccid, giữ mọi ccid phân biệt khác rỗng. |
| `bridge_exercise_problem` | 6.252.830 | Hợp ánh xạ relation và catalog problem, gộp theo exercise–problem, giữ provenance. |
| `bridge_lo_problem` | 461.180 | Nối exercise của chapter với problem; giữ max score và cờ assignment nhập nhằng. |
| `dim_lo_catalog` | 34.843 | Một course–chapter; đếm video/exercise phân biệt và problem phân biệt; các mẫu số chỉ số. |
| `bridge_user_course_enrollment` | 11.807.090 | Ghép `course_order[k]` với `enroll_time[k]`; lệch độ dài mảng vào quarantine, giữ cờ parse và trùng enrollment. |
| `bridge_course_comment` | 10.181.950 | Cặp course–comment; hỗ trợ gán course khi resource chưa có chapter chắc chắn. |
| `bridge_user_comment` | 8.395.141 | Cặp user–comment phục vụ provenance; pipeline nhãn hiện hành lấy tác giả từ bảng comment. |

Resource chapter gốc lỗi/không parse được được quarantine tại bước dim. Chương tổng hợp `course_id:CH-1` được tạo **sau đó** cho social event biết course nhưng chưa xác định duy nhất chapter; không biến nó thành chapter catalog có mẫu số giả.

### 3.3. Chuỗi kế thừa tới V4

| Bước | Code/nguồn | Biến đổi thực tế |
|---|---|---|
| Event mapping | [F04], [F05], [F06] | Problem chỉ giữ ánh xạ một `lo_key`; video gắn được một chapter. Video chỉ dùng timestamp trong `[2020-01-19,2020-11-04)`. Dựng bounds hoạt động của course. |
| Final V2 | [F15] → `final_quality_v2` | Cập nhật mapping social, mốc ngày, sessions, sentiment, P95; tính chỉ số chapter rồi q/y cuối kỳ. Dùng stage problem và video đã tạo ở bước trước. |
| Final V3 | [F18] → `final_quality_v3` | Loại cặp có đúng một chapter quan sát và chapter đó là `-1`; giữ nguyên scalar/q/y/reference của cặp còn lại. |
| Final V4 | [F21] + [FC] → `final_quality_v4` | Tạo family bằng connected components Jaccard ccid≥0,9, tối thiểu 5 ccid để xét gộp; chia theo family. Thêm metadata/hash ID; không tính lại P95 hay q/y. |
| Phase V2 | [F16] gọi [F09] → `phase_features_v2` | Lọc event theo cutoff từng phase; tính raw X, thêm cờ video và số chương `-1`, sequence. |
| Phase V3 | [F19] → `phase_features_v3` | Lọc cohort giống Final V3; giữ nguyên raw numeric/sequence V2, thay assignment/encoder/preprocessing theo biến thể V3. |
| Phase V4 | [F22] → `phase_features_v4` | Giữ nguyên 57 raw numeric và gender; thay 4 vocabulary, family split và metadata; fit lại một preprocessor chung trên labeled train P1–P4. |
| FullTrain-A V4 | [F20], [F22] | Feature cửa sổ λ=1 từ `fullA`; giữ đúng cohort train V4; thêm phase_fraction=1, dùng cùng encoder và preprocessor Pool B. |

## 4. Công thức các chỉ số tạo nhãn cuối kỳ

Công thức dưới đây được thực hiện chủ yếu ở [F15], các bước `user_video_components`, `video_p95_reference`, `chapter_raw`, `chapter_p95_reference`, `chapter_quality` và aggregate user–course. Năm SQL trọng yếu đã được so với checkpoint lịch sử, không chỉ đọc `formula_policy.json`.

Đặt `N_v,N_ex,N_p` là số video, exercise, problem **trong catalog của chapter**; `R` là các dòng problem quan sát trong `W_final`; `a_j` là attempts, `correct_j` là is_correct. Các count theo dòng chỉ đồng nghĩa count problem nếu tính duy nhất của nguồn còn đúng; code không khôi phục lịch sử từng attempt.

| Thành phần | Công thức thực tế | Mẫu số/reference và xử lý đặc biệt | Căn cứ |
|---|---|---|---|
| Thời gian video `s_uv` | `sum(end_point-start_point)` trên segment valid trong W_final, gồm xem lặp | Valid: điểm đầu≥0, cuối>đầu, điểm/speed hữu hạn, speed>0. Chọn theo **thời điểm bắt đầu** segment; nhánh nhãn này không dùng công thức clip đuôi của phase X. | [F06] staging, [F15] prepare |
| `TS_video` | Nếu `s_uv=0`: 0; còn lại `min(1,s_uv/P95+(s_uv))` | P95 theo **course–chapter–video**, trên user có tổng giây dương; không chia caption duration và không chia playback speed. | [F15], dòng 217; thích nghi từ QASOC §3.2.1 |
| `TS` chapter | `sum(TS_video)/N_v` khi `N_v>0` | Video catalog không xem đóng góp 0. Chapter không có catalog/mẫu số → NULL. Đây là chuẩn hóa từng video rồi mới tổng hợp. | [F15], `chapter_raw` |
| `RV` | `n_videos_watched/N_v` | Video được xem khi tổng content seconds>0; `N_v=0/NULL` → NULL. | [F15]; QASOC (5) |
| `AV` | `count_distinct(exercise_id đã làm)/N_ex` | `N_ex=0/NULL` → NULL. | [F15]; QASOC (7) |
| `IF_raw` | `sum(a_j)` | Chỉ khả dụng khi N_p>0 và không có attempts thiếu, ≤0, không hữu hạn hoặc không nguyên trong nhóm. Không log-transform. | [F15], `problem_*`, `chapter_raw`; QASOC (6) |
| `IF` | NULL nếu IF_raw NULL; 0 nếu IF_raw=0; còn lại `min(1,IF_raw/P95+(IF_raw))` | P95 theo **course–chapter**, trên IF_raw dương. | [F15], `chapter_p95_reference`, `chapter_quality` |
| `COELO` | `mean_available(TS,RV,AV,IF)` | Cần ít nhất **2/4** thành phần; có 0 hoặc 1 thành phần → NULL. Không ép mẫu số thành 4 khi thiếu. | [F15]; thích nghi QASOC (3) |
| `NVL` / `n_visits` | Số đầu phiên của từng user–course–chapter–resource | Phiên đầu hoặc gap giữa **hai event bắt đầu liên tiếp >30 phút**. Nguồn: video, problem, comment trước lọc text, reply authored; không cộng reply received. | [F15], `visits_*`; quy ước session của pipeline |
| `FV_raw` | `NVL/TNVL`, với `TNVL=sum_users(NVL)` | TNVL theo course–chapter của quần thể tham chiếu V2. | [F15]; QASOC (11) |
| `FV` | `min(1,FV_raw/P95+(FV_raw)) = min(1,NVL/P95+(NVL))` | Code dùng dạng rút gọn bên phải; P95 theo course–chapter. Tương đương khi TNVL là cùng hằng số dương. | [F15], `chapter_quality` |
| `FA_raw` | `(n_positive_comments-n_negative_comments)/n_comments_chapter_total` | Tử số đếm **argmax label** của comment user được inference. Mẫu số là **mọi comment của chapter** trong quần thể V2 sau chọn cửa sổ, trước lọc text/exact duplicate cho inference. Cần n_sentiment>0 và đủ prediction cho toàn bộ comment đủ điều kiện của user. | [F15], `chapter_comment_totals`, `chapter_raw`; thích nghi QASOC (12) |
| `FA` | Nếu FA_raw=NULL: NULL; nếu FA_raw=0: 0,5; còn lại `0.5+0.5*clip(FA_raw/P95abs(FA_raw),-1,1)` | P95 theo course–chapter trên **abs khác 0**; giữ dấu. FA_raw âm→dưới 0,5; dương→trên 0,5. | [F15], `chapter_quality` |
| `AFELO` | `mean_available(FV,FA)` | Có một thành phần vẫn tính; cả hai thiếu→NULL. Không dùng RS/rating, comment participation, reply participation hay response rate trong nhãn hiện hành. | [F15]; thích nghi QASOC (10) |
| `w(a)` | `max(0.1, 1-0.1*(a-1)) = max(1,11-a)/10` | Attempts phải là số nguyên dương, hữu hạn. Hàm giảm 0,1 mỗi lần, chặn ở 0,1 khi attempts≥10. | [F15], `correct_weight_tenths_sum`; mở rộng thang CFA QASOC |
| `KL` | `n_correct/N_p` | N_p>0 và không có correctness ngoài {0,1}/NULL; nếu không→NULL. | [F15]; QASOC (16) |
| `CFA` | `sum_correct(w(a_j))/n_correct` | n_correct>0 và attempts của mọi dòng đúng đều hợp lệ. Không có câu đúng→CFA NULL. **Không phải mean(1/attempts).** | [F15], `chapter_raw` |
| `ACELO` | `sum_correct(w(a_j))/N_p` | N_p>0, correctness hợp lệ và attempts của câu đúng hợp lệ. Có catalog nhưng không câu đúng→0. Tính trực tiếp để không biến `KL=0,CFA=NULL` thành điểm thiếu. | [F15]; thích nghi QASOC (15) |
| `LOQV_alpha` | `sqrt(sum_available(w_k*(1-I_k)^2)/sum_available(w_k))`, I=(COELO,AFELO,ACELO), w=(1,alpha,1) | Tính trên chiều khả dụng; một chiều cũng tính. Cả ba thiếu→NULL. Alpha chính=1; alpha 0,1 chỉ sensitivity. Điểm là khoảng cách: **nhỏ hơn gần vector lý tưởng hơn**. | [F15], `chapter_quality`; QASOC Algorithm 1, mở rộng trọng số/missing |
| `q_alpha` | `avg(LOQV_alpha)` trên chapter quan sát của user–course có điểm | Chapter biết và chapter -1 trong cặp mixed đều được tính; bỏ NULL, giữ 0; không chia cho toàn bộ chapter catalog chưa quan sát. Mean khoảng cách, không phải khoảng cách của mean vector. | [F15] aggregate; [F18] lọc cohort |
| `y_alpha` | E nếu q<0,2; G nếu 0,2≤q<0,4; A nếu q≥0,4 | q=NULL→y=NULL; không gán cặp chưa có nhãn thành A. Đây là ngưỡng của giao thức hiện tại. | [F15], [F21], `formula_policy.json` |
| `Q_c,n_c,Y_c` | `Q_c=mean(q_alpha_1)`; n_c=count enrollment có nhãn; Y_c áp cùng ngưỡng | Mỗi enrollment một lần, không đếm bốn phase thành bốn người; chỉ cohort có nhãn còn giữ. Không dùng các cột này làm X. | [F22], hàm `stats` |

**Quần thể P95:** Final V2 tính reference trên các enrollment đủ thời gian có event được mapping trong cửa sổ cuối kỳ, trước lọc V3 và trước family split V4; không lọc riêng `split='train'` trong SQL reference. V3/V4 giữ reference, kể cả ảnh hưởng của cặp sau đó bị loại hoặc được gán dev/test V4. Do đó không thể mô tả toàn bộ quá trình tạo target là “train-only V4”. Đây khác với việc không đọc test trong notebook thực nghiệm hiện tại.

**Nguồn học thuật cần phân biệt phiên bản:** [bài QASOC Abda và cộng sự, DOI 10.1007/s10639-023-12247-w](https://link.springer.com/article/10.1007/s10639-023-12247-w) là căn cứ chính; bản PDF cục bộ đã kiểm tra trực quan trang 7, 8, 10, 11, 12 và 13. Algorithm 1 ở trang 13 dùng `Distance/MaximumDistance`, phù hợp chiều khoảng cách của code alpha=1 khi đủ ba chiều. Kế hoạch cũ còn dẫn luận án năm 2025. Hồ sơ này xác định rõ bản bài báo năm 2024 đã đối chiếu; không mặc nhiên thay thế công thức giữa các tài liệu khác phiên bản.

## 5. P95 khác min–max thế nào?

| Nội dung | Min–max tham khảo | Cách đang dùng |
|---|---|---|
| Công thức dương | `(x-min)/(max-min)` | `min(1,x/P95+(x))`; zero hợp lệ giữ 0. |
| Mốc dưới | Dịch theo min của nhóm | Neo tại 0, không trừ min. |
| Mốc trên | Max của nhóm | Phân vị 95% các giá trị **dương**; từ P95 trở lên bão hòa ở 1. |
| Nhóm hiệu chuẩn | Phải khai báo nhóm tính min/max | TS: course–chapter–video; IF/FV/FA: course–chapter, quần thể V2 đã đóng băng. |
| Trường hợp FA có dấu | Min–max không bảo đảm giá trị 0 thành 0,5 | Chia cho P95 của abs khác 0, clip hai phía, rồi ánh xạ [-1,1]→[0,1]. |
| Nhóm chỉ có một trị dương | `max-min=0` nếu nhóm hằng | P95 bằng chính trị đó; điểm dương thành 1. FA âm/dương singleton thành 0/1. |
| Nhóm toàn 0 | Min–max cần chính sách mẫu số 0 | IF/FV zero→0; FA zero có evidence→0,5, dù P95 positive/nonzero NULL. |
| Dữ liệu thiếu | Cần chính sách riêng | NULL được xử lý trước phép chia; không tự thay NULL bằng 0 trong chỉ số nhãn. |
| Hệ quả | Nhạy với cực trị ở max/min | Giảm phụ thuộc max trong nhóm đủ lớn, nhưng nhóm nhỏ vẫn có thể bị cực trị chi phối; không chứng minh tính hợp lệ chỉ bằng thay scaler. |

Ví dụ minh họa có điều kiện: nếu một nhóm có min=0, max=1.000, P95=100 thì x=50 được min–max=0,05 nhưng P95=0,5; x=150 được P95=1. Đây là minh họa phép biến đổi, không phải thống kê đo từ bộ dữ liệu.

`quantile_cont` là phân vị nội suy liên tục, không đơn giản lấy dòng tại vị trí làm tròn 95%. Code không log IF hiện hành. FA dùng P95abs nên không thể áp công thức chuẩn hóa chỉ dành cho biến không âm.

Với `FA_raw=delta/M` và cùng hằng số M>0 trong một chapter, `P95(abs(delta/M))=P95(abs(delta))/M`; M triệt tiêu trong FA chuẩn hóa. Điều này chỉ đúng khi cùng tập user/độ khả dụng/reference. Thay mẫu số thô vẫn cần được khai báo, đặc biệt với chương -1 hoặc thay phạm vi comment.

## 6. Mô hình sentiment và quy trình inference

| Mục | Bản thực tế/nguồn chứng minh |
|---|---|
| Model | `cardiffnlp/twitter-roberta-base-sentiment-latest` - frozen pretrained RoBERTa ba lớp. |
| Revision khóa | `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`, cùng revision cho cache V1 và inference bổ sung V2. |
| Weight hash được ghi | `pytorch_model.bin`: `4d24a3e32a88ed1c4e5b789fc6644e2e767500554e954b27dccf52a8e762cbae`. Lần này đối chiếu lock, không tải/chạy lại trọng số. |
| Code | [F07] `preprocess`, `prepare_comments`; [F15] `infer`, `finalize`. |
| Preprocessing version | `cardiff-model-card-mentions-http-v1`: token bắt đầu @ và dài > 1→`@user`; token bắt đầu `http`→`http`; không thấy bước dịch bổ sung trong script. |
| Text đưa inference | Text gốc trong bảng **translated_comment**, sau chọn comment không rỗng và exact_duplicate_rank=1. `normalized_text` để lọc/dedup là trim + nén whitespace; inference gọi preprocess trên trường `text`. |
| Cache | SHA-256 của text **sau preprocessing**; inference một lần cho mỗi text, sau đó join prediction lại từng comment. Không gộp các comment khác nhau thành một lượt đóng góp chỉ vì trùng text cache. |
| Tokenizer | Tokenizer cùng snapshot; truncation=True, max_length=512 gồm special tokens, padding theo batch. Ghi n_tokens/was_truncated. |
| Đầu ra | Softmax logits float32; xác suất được chuẩn hóa lại; label là argmax. ID 0=negative, ID 1=neutral, ID 2=positive. |
| Lần cache ban đầu | Manifest ngày 17/10: CPU, batch 32, token_budget 2048, checkpoint 1024; torch 2.8.0, transformers 4.57.6, tokenizers 0.22.2, huggingface-hub 0.36.2, DuckDB 1.3.2, numpy 2.0.2. |
| Inference bổ sung V2 | `inference_config.json`: MPS, float32, threads 4, batch 24, checkpoint 512; tái sử dụng cache cũ. Không có package manifest riêng đầy đủ cho đợt bổ sung; không tự suy ra mọi version bằng bản cache. |
| Tham gia nhãn | FA_raw dùng **số comment positive/negative theo argmax**, neutral không tăng tử số. |
| Tham gia feature X | Ba feature sentiment dùng **mean xác suất** p_pos/p_neu/p_neg của comment trước cutoff, không dùng count argmax. |
| Reply | Dùng timestamp/quan hệ để tính hoạt động/visit/response; không suy luận sentiment riêng từ text reply. |
| Hạn chế | Model card mô tả mô hình tiếng Anh/Twitter. Chưa có đánh giá annotated riêng cho comment giáo dục; text CJK/mixed được giữ và gắn cờ. Prediction là proxy sentiment. |

Model card chính thức xác nhận cách thay mention/URL và các lớp: [Cardiff NLP](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest). Quy trình thực tế được khóa bởi [model_lock V2](final_quality_v2/sentiment/model_lock.json), [inference_config V2](final_quality_v2/sentiment/inference_config.json) và [manifest cache](afelo_sentiment_cardiff/run_manifest.json).

**Ba trường hợp FA cần tách:** không có comment được inference→FA=NULL; có evidence và toàn neutral→FA=0,5; positive và negative cân bằng cũng có FA_raw=0→FA=0,5. FA=0,5 không đồng nghĩa mọi comment đều neutral. AFELO vẫn có thể tính từ FV khi FA thiếu.

## 7. Bảng công thức chi tiết đủ 62 feature đầu vào

### 7.1. Định nghĩa tập event để đọc bảng

- `V_all`: video event đã mapping trong W_p, gồm invalid/rewind/zero-length. `V`: tập con valid và có giây nội dung quan sát>0.
- Với segment bắt đầu lúc t, đầu s, cuối b, speed r: `d=min(b-s,max(0,t_p-t)*r)`; đầu cuối quan sát `s+d`; kết thúc thời gian `t+d/r`. Invalid có d=0 và kết thúc tại t. **Không lật rewind thành đoạn xem.**
- `P`: dòng problem có submit_time trong W_p, ánh xạ một chapter. `C_all`: comment sau chuẩn ID/dedup ID, trước lọc text. `C`: C_all không rỗng và exact_duplicate_rank=1.
- `R_a`: reply tác giả trong W_p, có parent duy nhất và timestamp≥parent_time. `R_r`: reply tới comment thuộc C của user, có parent_time≤reply_time≤cutoff của **người nhận**; không lọc theo enrollment của người trả lời khi tính lượt nhận.
- `A`: `V_all UNION ALL P UNION ALL C_all UNION ALL R_a`, tức hoạt động riêng user. Social gồm C_all, R_a và R_r. Received không cộng lần nữa vào n_events/visit/chapter touched.
- Code gộp raw X: [F09] `aggregate_windows` + `feature_sql`, được [F16] ghi đè cửa sổ và activity. [F22] giữ nguyên raw numeric và fit lại encoder/scaler; không chạy inference sentiment theo phase.

Trong bảng, **0** là giá trị raw khi không có event của loại đó; **NULL** là chưa xác định về ratio/mean/recency. Tất cả 57 numeric sau đó cùng đi qua preprocessing ở mục 8. Các công thức raw X là đặc tả của pipeline hiện tại; không có bằng chứng 62 cột này là nguyên bộ feature đã công bố của TEMPO-DQ.

Các biến `course_n_*` và signature teacher/school/field lấy từ catalog toàn course, giả định catalog tĩnh và khả dụng khi dự đoán; không có bảng release-time của từng tài nguyên trong contract hiện tại. `score_valid` yêu cầu score hữu hạn, không âm, max_score>0 và score≤max_score+1e-9. `fallback_possible` yêu cầu is_correct thuộc {0,1}, max_score dương và hữu hạn. Đây là điều kiện cụ thể cho `problem_attempted_score_ratio`, khác điều kiện attempts/correctness dùng trong target academic.

| STT | Feature | Loại/nhóm | Công thức raw thực tế | Khi không có dữ liệu | Code |
|---:|---|---|---|---|---|
| 1 | `n_video_segments` | numeric / Video | count(V): segment valid có d>0 sau clip cutoff. | 0 | [F09], aggregate_windows |
| 2 | `n_videos_watched` | numeric / Video | count_distinct(video_id trong V). | 0 | [F09], aggregate_windows |
| 3 | `video_media_seconds` | numeric / Video | sum_V(d); giây nội dung gồm các lần xem lặp. | 0 | [F09], aggregate_windows |
| 4 | `video_wall_seconds` | numeric / Video | sum_V(d/speed); giây thời gian xem proxy, chưa khử phiên đồng thời. | 0 | [F09], aggregate_windows |
| 5 | `video_unique_seconds` | numeric / Video | Theo từng video, lấy độ dài hợp các khoảng [start_point,start_point+d], rồi cộng qua video. | 0 | [F09], aggregate_windows |
| 6 | `video_rewatch_seconds` | numeric / Video | max(0,video_media_seconds-video_unique_seconds). | 0 | [F09], aggregate_windows |
| 7 | `video_sessions_30m` | numeric / Video | Trong V, sort thời điểm bắt đầu trong user–course; count(first OR gap_start>30 phút). Đúng 30 phút chưa tách phiên. | 0 | [F09], aggregate_windows |
| 8 | `n_revisited_videos` | numeric / Video | Trong V, tách phiên riêng từng video bằng gap_start>30 phút; đếm video có hơn 1 phiên. | 0 | [F09], aggregate_windows |
| 9 | `video_mean_speed` | numeric / Video | avg_V(speed), trọng số mỗi segment bằng nhau; không weighted theo duration. | NULL nếu V rỗng | [F09], aggregate_windows |
| 10 | `video_mean_segment_seconds` | numeric / Video | avg_V(d) sau clip cutoff. | NULL nếu V rỗng | [F09], aggregate_windows |
| 11 | `video_active_days` | numeric / Video | count_distinct(DATE(event_time) trong V), dựa thời điểm bắt đầu segment. | 0 | [F09], aggregate_windows |
| 12 | `video_recency_days` | numeric / Video | (cutoff-max_V(observed_end_time))/86400. | NULL nếu V rỗng | [F09], aggregate_windows |
| 13 | `n_video_events` | numeric / Video | count(V_all), gồm cả video invalid/rewind/zero-length; không yêu cầu d>0. | 0 | [F16], video_flags |
| 14 | `n_rewind_segments` | numeric / Video | count_V_all(start_point>end_point AND finite(start_point) AND finite(end_point)); chỉ proxy rewind, không cộng giây xem. | 0 | [F16], video_flags |
| 15 | `n_invalid_video_segments` | numeric / Video | count_V_all(NOT valid_measurement), gồm sai khoảng hoặc speed không hợp lệ. | 0 | [F16], video_flags |
| 16 | `n_zero_length_segments` | numeric / Video | count_V_all(start_point=end_point); có thể đồng thời thuộc invalid. | 0 | [F16], video_flags |
| 17 | `n_problem_submissions` | numeric / Bài tập | count(P): số dòng trạng thái user–problem tại submit_time; không phải tổng từng attempt. | 0 | [F09], aggregate_windows |
| 18 | `n_exercises_attempted` | numeric / Bài tập | count_distinct(exercise_id trong P). | 0 | [F09], aggregate_windows |
| 19 | `problem_attempts` | numeric / Bài tập | sum_P(CASE WHEN attempts>0 THEN attempts ELSE 0 END). Không có kiểm nguyên/finite bổ sung như IF_raw của nhãn. | 0 | [F09], aggregate_windows |
| 20 | `n_correct_problems` | numeric / Bài tập | count_P(is_correct=1), đếm dòng; không dùng COUNT DISTINCT ở bước aggregate. | 0 | [F09], aggregate_windows |
| 21 | `n_first_try_correct` | numeric / Bài tập | count_P(is_correct=1 AND attempts=1). | 0 | [F09], aggregate_windows |
| 22 | `problem_correct_rate` | numeric / Bài tập | avg_P(is_correct::DOUBLE); SQL bỏ NULL, không tự lọc correctness ngoài {0,1} trong biểu thức avg. | NULL nếu không có giá trị | [F09], aggregate_windows |
| 23 | `problem_first_try_correct_rate` | numeric / Bài tập | n_first_try_correct/count(P); mẫu số gồm mọi dòng P. | NULL nếu P rỗng | [F09], aggregate_windows |
| 24 | `problem_mean_attempts` | numeric / Bài tập | avg_P(attempts WHERE attempts>0). | NULL nếu không có attempts dương | [F09], aggregate_windows |
| 25 | `problem_attempted_score_ratio` | numeric / Bài tập | sum(earned_j)/sum(problem_max_score_j). earned=score nếu score_valid; nếu không, is_correct*max_score khi fallback_possible. Chỉ tính nếu không có dòng unresolved. | NULL nếu P rỗng, mẫu số 0 hoặc có dòng không giải được score | [F04] valid flags; [F09] aggregate |
| 26 | `problem_efficiency_mean` | numeric / Bài tập | avg_P(is_correct/attempts WHERE efficiency_valid); flag yêu cầu attempts>0 và is_correct thuộc {0,1}. Không phải CFA của target. | NULL nếu không có dòng efficiency_valid | [F04], [F09] |
| 27 | `problem_active_days` | numeric / Bài tập | count_distinct(DATE(submit_time) trong P). | 0 | [F09], aggregate_windows |
| 28 | `problem_recency_days` | numeric / Bài tập | (cutoff-max_P(submit_time))/86400. | NULL nếu P rỗng | [F09], aggregate_windows |
| 29 | `n_comments` | numeric / Xã hội | count(C): comment text không rỗng, dedup exact user/resource/time/normalized_text; sau dedup ID. | 0 | [F09], aggregate_windows |
| 30 | `comment_mean_characters` | numeric / Xã hội | avg_C(length(text)); dùng text lưu, không phải số token hoặc length(model_text). | NULL nếu C rỗng | [F09], aggregate_windows |
| 31 | `n_replies_authored` | numeric / Xã hội | count(R_a), theo user_id của người trả lời và cửa sổ của người đó. | 0 | [F09], aggregate_windows |
| 32 | `n_replies_received` | numeric / Xã hội | count(R_r), số reply tới comment C của người học, được biết trước cutoff của người nhận. | 0 | [F09], aggregate_windows |
| 33 | `n_comments_with_reply` | numeric / Xã hội | count_distinct(comment_id trong R_r). | 0 | [F09], aggregate_windows |
| 34 | `comment_reply_rate` | numeric / Xã hội | coalesce(n_comments_with_reply,0)/n_comments. | 0 nếu có comment chưa có reply; NULL nếu không có comment | [F09], feature_sql |
| 35 | `sentiment_p_positive_mean` | numeric / Xã hội | avg_C(p_pos) sau LEFT JOIN prediction bằng comment_id; chỉ comment trước cutoff. | NULL khi không có prediction khả dụng | [F07], [F15], [F09] |
| 36 | `sentiment_p_neutral_mean` | numeric / Xã hội | avg_C(p_neu), giữ xác suất softmax, không biến thành count argmax. | NULL khi không có prediction khả dụng | [F07], [F15], [F09] |
| 37 | `sentiment_p_negative_mean` | numeric / Xã hội | avg_C(p_neg), giữ xác suất softmax, không dùng FA của target. | NULL khi không có prediction khả dụng | [F07], [F15], [F09] |
| 38 | `social_active_days` | numeric / Xã hội | count_distinct(DATE(event_time) trong C_all UNION ALL R_a UNION ALL R_r). Gồm comment trước lọc text. | 0 | [F16], activity_features |
| 39 | `social_recency_days` | numeric / Xã hội | (cutoff-max(event_time của C_all,R_a,R_r))/86400. | NULL nếu không có social event | [F16], activity_features |
| 40 | `n_events` | numeric / Hoạt động | count(A)=count(V_all)+count(P)+count(C_all)+count(R_a); không cộng lượt nhận reply. | 0 | [F16], activity_features |
| 41 | `n_active_days` | numeric / Hoạt động | count_distinct(DATE(event_time) trong A). | 0 | [F16], activity_features |
| 42 | `n_chapters_touched` | numeric / Hoạt động | count_distinct(lo_key trong A WHERE lo_key NOT LIKE %:CH-1). | 0 | [F16], activity_features |
| 43 | `activity_span_days` | numeric / Hoạt động | (max_A(observed_end_time)-min_A(event_time))/86400; video dùng đuôi đã clip, event khác dùng timestamp của nó. | NULL nếu A rỗng; event điểm duy nhất có thể 0 | [F16], activity_features |
| 44 | `activity_recency_days` | numeric / Hoạt động | (cutoff-max_A(observed_end_time))/86400. | NULL nếu A rỗng | [F16], activity_features |
| 45 | `n_visits_30m` | numeric / Hoạt động | Trong A, partition user–course–chapter–resource; count(first OR gap_start>30 phút). Problem dùng exercise_id; social thiếu resource dùng discussion:course_id. | 0 | [F16], visit_features |
| 46 | `n_unknown_chapters_touched` | numeric / Hoạt động | count_distinct(lo_key trong A WHERE lo_key LIKE %:CH-1); tối đa 1 trong mỗi user–course. | 0 | [F16], activity_features |
| 47 | `n_chapters_touched_all` | numeric / Hoạt động | count_distinct(lo_key trong A), gồm chapter -1. | 0 | [F16], activity_features |
| 48 | `course_n_videos` | numeric / Ngữ cảnh | count_distinct(resource_id WHERE resource_type=video), theo course trong dim_lo_resource. | 0 khi catalog course không có video | [F02]; [F09], prepare_static |
| 49 | `course_n_exercises` | numeric / Ngữ cảnh | count_distinct(resource_id WHERE resource_type=exercise), theo course. | 0 khi catalog course không có exercise | [F02]; [F09], prepare_static |
| 50 | `course_n_problems` | numeric / Ngữ cảnh | count_distinct(problem_id) theo course trong bridge_lo_problem. | 0 khi không có problem mapping | [F02]; [F09], prepare_static |
| 51 | `course_n_chapters` | numeric / Ngữ cảnh | count(*) theo course trong dim_lo_catalog; không cộng chapter -1 tổng hợp. | 0 khi không có chapter catalog | [F02]; [F09], prepare_static |
| 52 | `learner_age_at_enrollment` | numeric / Ngữ cảnh | year(enroll_time)-year_of_birth, chỉ giữ birth≥1900 và chênh năm trong [10,100]; không tính tuổi chính xác theo ngày sinh. | NULL nếu thiếu/ngoài miền | [F16], build_windows |
| 53 | `phase_fraction` | numeric / Ngữ cảnh | P1=0.25,P2=0.50,P3=0.75,P4=0.90; FullTrain-A=1.0. Numeric z-score chung; bỏ khỏi selector. | Luôn có theo phase | [F09], feature_sql; [F22] |
| 54 | `video_coverage_ratio` | numeric / Ngữ cảnh | coalesce(n_videos_watched,0)/course_n_videos. | 0 nếu chưa xem và mẫu số>0; NULL nếu mẫu số 0/thiếu | [F09], feature_sql |
| 55 | `exercise_coverage_ratio` | numeric / Ngữ cảnh | coalesce(n_exercises_attempted,0)/course_n_exercises. | 0 nếu chưa làm và mẫu số>0; NULL nếu mẫu số 0/thiếu | [F09], feature_sql |
| 56 | `problem_coverage_ratio` | numeric / Ngữ cảnh | coalesce(n_problem_submissions,0)/course_n_problems; tử số đếm dòng, không count_distinct(problem_id) tại đây. | 0 nếu chưa làm và mẫu số>0; NULL nếu mẫu số 0/thiếu | [F09], feature_sql |
| 57 | `chapter_coverage_ratio` | numeric / Ngữ cảnh | coalesce(n_chapters_touched,0)/course_n_chapters, chỉ chapter biết; code không clip ratio. | 0 nếu chưa chạm và mẫu số>0; NULL nếu mẫu số 0/thiếu | [F09], feature_sql |
| 58 | `learner_code` | categorical | Token=user_id; vocabulary từ labeled train V4, sort token→row_number bắt đầu 1. | Unseen/thiếu→0 | [F22], encoder_sources |
| 59 | `teacher_set_code` | categorical | Token=signature tập DISTINCT teacher_id được sort theo course; vocabulary chỉ từ course có enrollment train. | Signature thiếu/unseen→0 | [F09] static; [F22] encoder |
| 60 | `school_set_code` | categorical | Token=signature tập DISTINCT school_id được sort theo course; vocabulary train-only. | Signature thiếu/unseen→0 | [F09] static; [F22] encoder |
| 61 | `field_set_code` | categorical | Token=list_sort(field) nối thành signature, rỗng→NULL; vocabulary train-only. Không có DISTINCT bổ sung ở biểu thức field signature. | Signature thiếu/unseen→0 | [F09] static; [F22] encoder |
| 62 | `gender_code` | categorical | Giữ gender nguồn khi thuộc {0,1,2}; ngoài miền hoặc thiếu→-1; category, không z-score. | -1 | [F09] static; [F22] giữ nguyên |

### 7.2. Weekly sequence

`week_index=floor((event_time-e)/(7*86400))`, bắt đầu từ0. Mỗi tuần cộng 11 trường: n_video_segments, video_media_seconds, n_problem_submissions, problem_attempts, n_correct_problems, n_comments, n_replies_authored, n_replies_received, n_video_events, n_rewind_segments, n_invalid_video_segments. Trường không thuộc loại event đó đóng góp0.

Segment video được gán theo **tuần bắt đầu**, không chia giây qua nhiều tuần. Không tạo tuần trống; enrollment không có event có sequence rỗng. Weekly n_correct_problems là tổng indicator `(is_correct=1)::INT`; giữ hành vi NULL của SQL nếu trường nguồn thiếu. Sequence là đầu vào riêng, không cộng thêm 11 trường này vào inventory62 của tabular X. [F16].

## 8. Tiền xử lý X của V4: khác với P95 của target

Đặt T_train là hợp labeled train P1–P4: **131.906 enrollment ×4 =527.624 snapshot**. Với mỗi numeric j:

```text
median_j = median(non_NULL raw_j trên T_train)
imputed_j = coalesce(raw_j, median_j)
mean_j = avg(imputed_j trên T_train)
std_j = stddev_pop(imputed_j trên T_train)
scale_j = std_j nếu finite và >1e-12, ngược lại 1
ready_j = (imputed_j - mean_j) / scale_j
```

Cùng median/mean/scale áp cho mọi phase và FullTrain-A; không fit riêng A hoặc dev. Không thêm missing indicator; hiện không có numeric all-missing train bị loại. Categorical giữ dạng mã; không z-score chúng. Một feature raw tỷ lệ trong[0,1] có thể ra ngoài[0,1] sau z-score; đây là hành vi bình thường.

Ví dụ `phase_fraction`: median=0,625; mean≈0,6; scale≈0,2474873734. Pooled train giữ được bốn giá trị phase khác nhau; FullTrain-A raw=1 dùng cùng phép biến đổi, không bị chuẩn hóa bằng scaler A riêng.

### 8.1. Tham số đã lưu cho từng numeric

Nguồn: [preprocessing.json](phase_features_v4/preprocessing.json). Bảng hiển thị 12 chữ số có nghĩa cho dễ đọc; JSON giữ độ chính xác đầy đủ. Tái tính độc lập trên train khớp trong sai số số thực tương đối1e-8.

| Numeric | Median train | Mean sau impute | Scale dùng transform |
|---|---:|---:|---:|
| `n_video_segments` | 0 | 2.22979432323 | 25.9193276694 |
| `n_videos_watched` | 0 | 0.527610571164 | 2.43044329014 |
| `video_media_seconds` | 0 | 168.860254582 | 918.129637633 |
| `video_wall_seconds` | 0 | 140.490712439 | 735.425273137 |
| `video_unique_seconds` | 0 | 163.807737176 | 884.829263832 |
| `video_rewatch_seconds` | 0 | 5.05251740682 | 74.4368466884 |
| `video_sessions_30m` | 0 | 0.25136460813 | 0.90526307279 |
| `n_revisited_videos` | 0 | 0.032020908829 | 0.286603979484 |
| `video_mean_speed` | 1 | 1.02373538643 | 0.130739047429 |
| `video_mean_segment_seconds` | 76.8805263158 | 81.8457200161 | 45.4542832319 |
| `video_active_days` | 0 | 0.185220914894 | 0.577079788706 |
| `video_recency_days` | 38.1950671296 | 38.597091235 | 9.52478193851 |
| `n_video_events` | 0 | 2.24496421694 | 26.0376585687 |
| `n_rewind_segments` | 0 | 0.0143075371856 | 0.567787786459 |
| `n_invalid_video_segments` | 0 | 0.0151698937122 | 0.577415778241 |
| `n_zero_length_segments` | 0 | 0.000862356526617 | 0.0388073595013 |
| `n_problem_submissions` | 14 | 68.5907388595 | 100.162344925 |
| `n_exercises_attempted` | 2 | 12.8961684836 | 21.5747856714 |
| `problem_attempts` | 14 | 73.465945446 | 109.606766875 |
| `n_correct_problems` | 9 | 57.5252130305 | 90.5198064368 |
| `n_first_try_correct` | 9 | 55.1090454566 | 87.4953915219 |
| `problem_correct_rate` | 0.82196969697 | 0.772936988837 | 0.211872993913 |
| `problem_first_try_correct_rate` | 0.784090909091 | 0.743021950397 | 0.212132618714 |
| `problem_mean_attempts` | 1 | 1.03801860836 | 0.153451928711 |
| `problem_attempted_score_ratio` | 0.75 | 0.734656831975 | 0.151722788828 |
| `problem_efficiency_mean` | 0.8 | 0.756376377251 | 0.209538071128 |
| `problem_active_days` | 1 | 1.78989962549 | 2.5496099394 |
| `problem_recency_days` | 51.9570862269 | 57.1276044971 | 36.2031331592 |
| `n_comments` | 0 | 0.706952299365 | 2.73213998675 |
| `comment_mean_characters` | 26.7142857143 | 33.3842496559 | 63.3120253968 |
| `n_replies_authored` | 0 | 0.036493791033 | 2.41493110493 |
| `n_replies_received` | 0 | 0.0455305293163 | 0.728909251746 |
| `n_comments_with_reply` | 0 | 0.0358342304368 | 0.608585580332 |
| `comment_reply_rate` | 0 | 0.0126437380017 | 0.104787317924 |
| `sentiment_p_positive_mean` | 0.32149538278 | 0.339380905118 | 0.146436094453 |
| `sentiment_p_neutral_mean` | 0.532782775047 | 0.524179015537 | 0.126283289316 |
| `sentiment_p_negative_mean` | 0.054382786098 | 0.0645218077787 | 0.0726473150106 |
| `social_active_days` | 0 | 0.385460100375 | 1.05331642424 |
| `social_recency_days` | 92.8380821759 | 96.2914158204 | 35.9377967887 |
| `n_events` | 17 | 71.5791643291 | 103.478106614 |
| `n_active_days` | 1 | 2.24124945037 | 2.57665470477 |
| `n_chapters_touched` | 2 | 4.00314997043 | 4.27714178292 |
| `activity_span_days` | 0.566643518519 | 8.98931813037 | 19.7105846024 |
| `activity_recency_days` | 55.8475868056 | 66.4480702428 | 52.8679126784 |
| `n_visits_30m` | 4 | 14.4952447197 | 22.1221528975 |
| `n_unknown_chapters_touched` | 0 | 0.011809546192 | 0.108028148233 |
| `n_chapters_touched_all` | 2 | 4.01495951663 | 4.27531311235 |
| `course_n_videos` | 60 | 67.6555880703 | 56.9796006031 |
| `course_n_exercises` | 22 | 34.1099646718 | 30.0175876711 |
| `course_n_problems` | 186 | 210.088563068 | 162.749031898 |
| `course_n_chapters` | 10 | 10.243764499 | 4.59116196465 |
| `learner_age_at_enrollment` | 26 | 26.0018043152 | 0.21479213569 |
| `phase_fraction` | 0.625 | 0.599999999999 | 0.247487373415 |
| `video_coverage_ratio` | 0 | 0.00994067538174 | 0.0490422594809 |
| `exercise_coverage_ratio` | 0.125 | 0.333803231128 | 0.385969694466 |
| `problem_coverage_ratio` | 0.1 | 0.319675677414 | 0.380174067395 |
| `chapter_coverage_ratio` | 0.230769230769 | 0.399974299818 | 0.367131051095 |

### 8.2. Vocabulary categorical

`learner/teacher_set/school_set/field_set`: tập token phân biệt từ **labeled train V4**, sort token rồi `row_number()` bắt đầu 1; unseen/missing=0. Signature teacher/school là tập ID phân biệt đã sort; field là list field đã sort, nối bằng ký tự phân cách. Gender giữ 0/1/2 hợp lệ, ngoài miền/thiếu=-1; **quy tắc unknown=0 của encoder không áp thay cho gender**.

Tái kiểm tập token: learner **126.016**, teacher_set **1.162**, school_set **320**, field_set **65**; 0 token lệch tập train. Mã categorical không biểu diễn khoảng cách có ý nghĩa; selector không dùng chúng. [F22], dòng 174.

## 9. NULL, giá trị 0 và chương -1

| Tình huống | Quy tắc hiện hành | Hệ quả cần diễn giải |
|---|---|---|
| Chưa có event ở phase sớm | Giữ enrollment có target cuối kỳ; count/sum activity=0, mean/recency không xác định=NULL | Không loại người học chỉ vì không có hoạt động sớm. |
| Không có bài đã làm | Count problem=0; mean/rate raw X=NULL; coverage=0 nếu catalog denominator>0 | Không đồng nhất “không làm” với quan sát trả lời sai. |
| Không có comment | n_comments=0; mean probabilities và comment_reply_rate=NULL | Không gán neutral 0,5 cho sentiment X khi không có evidence. |
| Có comment nhưng chưa có reply | n_comments_with_reply=0, comment_reply_rate=0 | 0 là quan sát hợp lệ. |
| Thiếu modality của nhãn | Mean trên chiều khả dụng với coverage rule riêng: COELO có ít nhất 2 thành phần; AFELO có ít nhất 1 thành phần; LOQV có ít nhất 1 thành phần | Không tự điền chỉ số nhãn thiếu bằng 0. |
| Câu đúng không có | CFA=NULL nhưng ACELO=0 nếu catalog/correctness đủ điều kiện | Dùng công thức ACELO trực tiếp, giữ kết quả0. |
| Catalog không có/mẫu số 0 | Ratio hoặc chỉ số phụ thuộc catalog=NULL | `NULLIF(denominator,0)`; không gán mẫu số 1. |
| Không xác định chapter nhưng xác định course cho comment | `lo_key=course_id:CH-1`, chapter_major=-1 | Không suy ra chapter theo thứ tự thời gian hoặc chọn tùy ý. |
| Reply gắn được parent | Kế thừa course/chapter/resource của parent; parent mapping phải duy nhất; reply không sớm hơn parent | Không suy luận chapter độc lập từ text reply. |
| Chỉ quan sát chapter -1 | V3/V4 loại cả cặp khi n_chapters_observed=1 và n_unknown_chapters=1 | Không loại mọi cặp có xuất hiện -1. |
| Có cả chapter biết và -1 | Giữ cả hai trong chapter_details và mean q nếu có điểm | Chương -1 thường có FV/FA, thiếu mẫu số TS/RV/AV/KL/ACELO; điểm có thể dựa trên ít modality hơn. |
| Feature chapter coverage | `n_chapters_touched` bỏ -1; hai biến unknown/all ghi riêng | Không cộng -1 vào mẫu số catalog, không gọi nó là chapter nội dung xác định. |
| Enrollment chưa có nhãn | Giữ kho đầy đủ; q/y/split/enrollment_id/row_id theo contract có thể NULL, không vào supervised | Không biến hơn 10 triệu cặp chưa có nhãn thành class A. |
| Model-ready X | Điền median train rồi z-score | NULL raw không còn là NULL trong 57 numeric ready; giá trị điền không phải quan sát thật. |

## 10. Những điều chỉnh cần công khai trong O24

| Nội dung | Đang thực hiện | Căn cứ/giới hạn cần ghi |
|---|---|---|
| Đơn vị LO | Big chapter lấy thành phần số đầu | Quy ước mapping dataset tại notebook 02; không phải từng resource là một LO. |
| TS | P95 theo video, sau đó mean theo số video catalog; gồm giây xem lặp | Thích nghi từ thời gian tương tác; không dùng duration proxy làm thước đo hoàn thành video. |
| IF | Tổng attempts của trạng thái problem có submit_time quan sát được, chuẩn P95 | Không phải đếm riêng từng interaction theo timestamp attempt; không log IF. |
| Chuẩn hóa | P95 dương, FA signed-P95 | Thay min–max QASOC; là lựa chọn triển khai đã ghi policy, chưa có thẩm định riêng chỉ từ việc chọn P95. |
| FV | Phiên user–resource gap>30 phút; chuẩn P95 sau tỷ phần visits | Quy ước session của pipeline, không phải một click mặc nhiên là một visit. |
| FA | Sentiment comment làm proxy positive/negative; mẫu số mọi comment chapter | Thay explicit appreciation; model English/Twitter và dữ liệu dịch có giới hạn miền. |
| AFELO | Mean FV/FA khả dụng; bỏ RS | Input không cung cấp rating tương đương để đưa vào công thức hiện tại. |
| CFA/ACELO | Penalty từng problem đúng, tổng hợp theo catalog; floor 0,1 cho attempts lớn | Mở rộng beyond thang 5 attempt của QASOC; khác feature `problem_efficiency_mean=mean(correct/attempts)`. |
| Missing | Chấp nhận số chiều khác nhau, COELO có coverage ≥ 2 | Các điểm có cùng giá trị có thể dựa trên số nguồn bằng chứng khác nhau; cần báo coverage. |
| Alpha | Chính=1; alpha 0,1 sensitivity | Chỉ đủ ba chiều và alpha 1 mới đồng nhất khoảng cách Euclid chuẩn hóa ba chiều của bài QASOC. |
| Target user–course | Mean chapter có điểm của từng người | Không phải nhãn chuyên gia; khác bước tổng hợp population/chapter/course của nguồn tham khảo. |
| Nhãn E/G/A | Ngưỡng distance 0,2 và 0,4 của giao thức hiện tại | Không gán ba ngưỡng này là toàn bộ hệ phân loại của tài liệu nguồn. |
| Reference target | Đóng băng V2, không refit sau lọc V3/split V4 | Không tuyên bố P95 target fit chỉ trên V4 train. |
| Time | Enrollment đầu ngày; course_end cuối ngày từ last activity hồi cứu | Không phải lịch vận hành online đã biết trước; không có sửa offset múi giờ nguồn. |
| Video ở final so với phase | Nhãn chọn full segment theo start_time; phase X clip nội dung tại cutoff | Hai nhánh có quy tắc biên khác nhau; không ghi “mọi total video đều clip cùng một cách”. |
| Bộ X | 57 numeric+5 categorical từ raw-log baseline | Căn cứ trực tiếp là code/dictionary hiện có. Chưa phải mapping/tái lập đầy đủ feature TEMPO-DQ. |
| Chia tập | V4 group theo family nội dung, test là họ mở muộn | Thay assignment V3; raw numeric, cutoff và target không đổi. |
| Scaler X | Một pool train bốn phase; encoder train-only | Thay preprocessing per-phase V2/V3; việc fit X train-only không sửa phạm vi reference của nhãn. |

**Các điểm còn cần xác nhận:** căn cứ chấp nhận các thích nghi công thức/coverage; provenance của bước dịch; chênh hash script V2; ảnh hưởng reference V2; độ phù hợp sentiment với miền giáo dục; mapping TEMPO-DQ theo yêu cầu review. Tài liệu này không tự đóng các ô D06/O24 trong checklist.

## 11. Phiên bản code và môi trường

Repo cục bộ đang ở nhánh `main` **chưa có commit**, nên không thể cung cấp Git commit có thật cho các file này. Version được truy vết bằng version string, hash hiện tại và hash manifest đã ghi.

| Code | SHA-256 hiện tại | Đối chiếu manifest |
|---|---|---|
| [15_build_final_quality_v2.py](15_build_final_quality_v2.py) | `0eaa181c674ec3c8214c6ad27c6df6aa1fe71d21fe0dd9d0772f7b636f8044c2` | **Khác hash đã ghi** |
| [16_build_phase_features_v2.py](16_build_phase_features_v2.py) | `178ed4d1d0b9fa287f3ffbdf58efba16d665cbcced6f9ef8b2b6b44c8db1d6f0` | Khớp |
| [18_build_final_quality_v3.py](18_build_final_quality_v3.py) | `952781027fbd2e1ab9cb6647fe5c289c66b44fcf95d9842736ceeca00e1b82b1` | Khớp |
| [19_build_phase_features_v3.py](19_build_phase_features_v3.py) | `373de0f98c1a829f79e5f4ec0db9387a666c1fab877609ab837ae87000a2ff85` | Khớp |
| [20_build_fullA.py](20_build_fullA.py) | `b23ab06c7bc10a0980400463c481c7ffe3bec53a4a8658959d8af7eed540673c` | Khớp |
| [21_build_final_quality_v4.py](21_build_final_quality_v4.py) | `89be3aefd8ac9afa18a1ea3955e33c1a8c2b05bbaceb4131d942178894cada86` | Khớp |
| [22_build_phase_features_v4.py](22_build_phase_features_v4.py) | `3aa7316ee0bd5b704a1da6c0978f3a7f5030843b106100665c832b66c30d83c1` | Khớp |
| [cq_v4_common.py](cq_v4_common.py) | `53b85faebed9106986b1eaa2d79acefebac30ca431a8fe9bac5e8ce2ee4df164` | Khớp |
| [07_compute_afelo_sentiment.py](07_compute_afelo_sentiment.py) | `c8c32661b2e91c81d43a91fc519394c2b6bb112cded381146090e36f0ca6150f` | Khớp |
| [09_build_phase_features.py](09_build_phase_features.py) | `84993cb4153a15409af82df3151bfd59f606028db86a8b397a747d83054dbe92` | Chỉ ghi hash hiện tại; chưa có hash lần chạy tương ứng trong bảng đối chiếu |
| [04_eda_engagement_features_comparison.py](04_eda_engagement_features_comparison.py) | `ed7a3cd775e4199204d773a978f5598ca7c7fe3c8f11ac11ef511e2597c823b3` | Chỉ ghi hash hiện tại; chưa có hash lần chạy tương ứng trong bảng đối chiếu |
| [05_build_review_time_phases.py](05_build_review_time_phases.py) | `7984e2a5318f58100cedcbea59961bab726be9d01503111d13703d1d243a735c` | Chỉ ghi hash hiện tại; chưa có hash lần chạy tương ứng trong bảng đối chiếu |
| [06_build_final_quality_tables.py](06_build_final_quality_tables.py) | `c94182c2b0c0cff1537d138e0ec071301c246552190801e9f4017a5241fd1211` | Chỉ ghi hash hiện tại; chưa có hash lần chạy tương ứng trong bảng đối chiếu |
| [01_moocubex_raw_to_parquet.ipynb](01_moocubex_raw_to_parquet.ipynb) | `b55ba94ba4fa55ab94d3fe82cb5d999697932189601c974025981a03c97a42d9` | Chỉ ghi hash hiện tại; chưa có hash lần chạy tương ứng trong bảng đối chiếu |
| [02_build_dimensions_bridges.ipynb](02_build_dimensions_bridges.ipynb) | `9d957b1a681dfb1bf09ae1817dab125ad7f52d686ca48a9e05a48acc4c6241ba` | Chỉ ghi hash hiện tại; chưa có hash lần chạy tương ứng trong bảng đối chiếu |

Builder Final/Phase V4 ghi DuckDB 1.5.5, numpy 2.4.6, pandas 3.0.3. `requirements-v4.txt` hướng dẫn Python 3.12 và pytest 8.4.2, nhưng riêng run manifest builder không ghi Python version; không đồng nhất môi trường đề nghị với mọi lần chạy lịch sử. Audit mới dùng Python/DuckDB được ghi trong evidence.json; chỉ đọc và tính lại train/dev, không rebuild dữ liệu.

**Chênh hash V2:** hash manifest `5ab5a8f62bc94070cf34949617b28a04e6e3a61158c1b518e3bb2be590fbdd48`; hash file hiện tại `0eaa181c674ec3c8214c6ad27c6df6aa1fe71d21fe0dd9d0772f7b636f8044c2`. Đối chiếu 5 SQL không chứng minh mọi phần còn lại không đổi; cần bản script gốc hoặc lịch sử thay đổi để đóng hoàn toàn provenance code V2.

## 12. Kiểm tra trong lần lập tài liệu và cách tái kiểm

| Kiểm tra | Kết quả/phạm vi |
|---|---|
| Feature inventory | 62 tên trong dictionary khớp allowlist learner; 57 numeric có đủ tham số preprocessing. |
| Tham số preprocessing | Tái tính median, mean sau impute và stddev_pop từ 527.624 train snapshot;57/57 khớp. |
| Raw→ready | P1–P4 × train/dev: mỗi phase 131.906 train và 22.717 dev;8/8 bảng có 0 mismatch. |
| Vocabulary | Tập token của 4 encoder đúng labeled train; không xuất token/ID vào báo cáo. |
| SQL tạo nhãn | `video_p95_reference`, `user_video_ts`, `chapter_raw`, `chapter_p95_reference`, `chapter_quality`: 5/5 exact match giữa code hiện tại và checkpoint. |
| Hash builder | So với manifest từng bước; giữ phát hiện chênh script 15 thay vì ghi tất cả khớp. |
| Nguồn học thuật | Đối chiếu bài QASOC cục bộ và trang xuất bản; model card Cardiff; schema user chính thức MOOCCubeX. |
| Phạm vi chưa kiểm | Chưa dựng lại toàn bộ pipeline; chưa tái suy luận sentiment; chưa kiểm thủ công 20 ca; không đọc feature/label từng dòng test. |

Script đọc/kiểm: [audit_formula_feature_v4.py](formula_feature_v4_audit/audit_formula_feature_v4.py). Chạy từ workspace với môi trường có DuckDB:

```bash
python formula_feature_v4_audit/audit_formula_feature_v4.py
```

Script chỉ ghi `formula_feature_v4_audit/evidence.json`, không sửa dữ liệu hay checksum gói cũ. `validation_pass=true` trong evidence chỉ áp dụng các kiểm tra đã nêu, không phải phê duyệt M0 hay thẩm định khoa học.

## 13. Nguồn và vị trí tra công thức

| Mã | Nguồn trực tiếp | Vị trí cần đọc |
|---|---|---|
| F01 | [01 raw→Parquet][F01] | Cell code index 3 schema, cell 6 chuyển định dạng; đối chiếu manifest input. |
| F02 | [02 dim/bridge][F02] | Cell 8 course/problem/video; cell 9 resource/chapter; cell 11 relations; cell 13 catalog/enrollment; cell 15 comment bridges. |
| F04 | [04 mapping problem][F04] | Dòng 111–125 mapping; 176–184 stage_query và score_valid/fallback/efficiency. |
| F05 | [05 phase times][F05] | cutoff_sql/time_status_sql và course_source_time_bounds/course_time_bounds. |
| F06 | [06 final staging][F06] | Dòng214–223 video_events/valid_measurement và cửa sổ nguồn video. |
| F07 | [07 sentiment][F07] | Dòng 24 model constants;48 preprocess;85 prepare_comments; inference/cache. |
| F09 | [09 raw feature formulas][F09] | Dòng 116 prepare_static;189 aggregate_windows;262 feature_sql. |
| F15 | [15 Final V2][F15] | Prepare: mapping/time/visits;217 P95 video;387 chapter_raw;406 reference;414 chapter_quality; aggregate q/y. |
| F16 | [16 Phase V2][F16] | Dòng 46 build_windows;78 aggregate_windows ghi đè activity; gọi công thức F09. |
| F18 | [18 Final V3][F18] | EXCLUDE và giữ reference/scalar. |
| F19 | [19 Phase V3][F19] | Giữ raw V2, đổi cohort/assignment/encoder. |
| F20 | [20 FullA][F20] | Cửa sổ λ=1 và gọi bộ aggregate raw feature. |
| F21 | [21 Final V4][F21] | family/split, copy giá trị nhãn và reference. |
| F22 | [22 Phase V4][F22] | Dòng 174 vocabulary;231 fit train pool;246 make_ready; stats cho Qc/nc. |
| FC | [cq_v4_common][FC] | VERSION/DATE;course_families;family_split;hash helpers. |

Các mã F là **căn cứ triển khai**; chúng không thay thế tài liệu khoa học làm căn cứ chọn chỉ số.

- Abda, Y., Mehenaoui, Z., Lafifi, Y., Boudjehem, R. *A new approach for assessing the quality of online courses*. Education and Information Technologies 29, 13929–13956 (2024), online 28/12/2023. [Trang xuất bản và DOI](https://link.springer.com/article/10.1007/s10639-023-12247-w). [PDF cục bộ](../01_ly_thuyet/02_bai_bao_QASOC_Abda_2024.pdf). Tra §3.2.1, §3.2.2, §3.2.3 và Algorithm 1. Đây là bài tham khảo công thức; P95, session 30 phút, sentiment, missing và cohort là các thích nghi được ghi riêng ở trên.
- [Model card Cardiff NLP](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest): model tiếng Anh, preprocessing mention/URL và label mapping; revision thực chạy lấy từ lock cục bộ.
- [Schema user/behavior chính thức MOOCCubeX](https://github.com/THU-KEG/MOOCCubeX/blob/main/docs/user-en.md): trường enrollment, comment/reply, user-problem và user-video. Schema mô tả nguồn, không quy định công thức feature 62 cột của dự án.
- [Kế hoạch công thức cũ](02_plan_code_tao_nhan_moocubex.md) và [tài liệu schema/chỉ số nội bộ](moocubex_schema_va_danh_gia_3_chi_so.md) có nhiều phần lịch sử/proxy. Khi khác công thức hiện hành, ưu tiên SQL checkpoint và code đã đối chiếu; không lấy công thức AFELO_proxy lịch sử thay AFELO V4.
- [Formula policy V4](final_quality_v4/formula_policy.json), [feature columns](phase_features_v4/feature_columns.json), [feature dictionary](phase_features_v4/feature_dictionary.json), [preprocessing](phase_features_v4/preprocessing.json), [manifest Final V4](final_quality_v4/run_manifest.json), [manifest Phase V4](phase_features_v4/run_manifest.json).

[F01]: 01_moocubex_raw_to_parquet.ipynb
[F02]: 02_build_dimensions_bridges.ipynb
[F04]: 04_eda_engagement_features_comparison.py
[F05]: 05_build_review_time_phases.py
[F06]: 06_build_final_quality_tables.py
[F07]: 07_compute_afelo_sentiment.py
[F09]: 09_build_phase_features.py
[F15]: 15_build_final_quality_v2.py
[F16]: 16_build_phase_features_v2.py
[F18]: 18_build_final_quality_v3.py
[F19]: 19_build_phase_features_v3.py
[F20]: 20_build_fullA.py
[F21]: 21_build_final_quality_v4.py
[F22]: 22_build_phase_features_v4.py
[FC]: cq_v4_common.py
