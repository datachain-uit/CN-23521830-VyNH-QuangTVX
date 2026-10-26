# Y17 — danh mục tài liệu dùng cho giao thức CQ

Metadata đối chiếu nguồn chính ngày 25/10/2026. Danh mục ở [`references.bib`](../references.bib); bảng này ghi phạm vi sử dụng và phần cần kiểm tiếp, không thay bài tổng quan hoặc kiểm tra tính mới.

| Nhóm yêu cầu | Tài liệu / khóa BibTeX | Vai trò và lưu ý |
|---|---|---|
| Baseline bảng | `hadar2024coretab` | CoreTab; trong panel CQ dùng bản XGB OVA adapted, phải ghi khác native binary |
| Baseline bảng | `chen2022bdis` | BDIS; cite đúng phương pháp và giữ khóa mã nguồn |
| Baseline bảng | `florea2026tabularsource` | Nguồn benchmark GCoreSet đang ghi under review; không gán venue/DOI đã xuất bản. Tên tác giả giữ initials theo README |
| Baseline chọn dữ liệu | `xia2023moderate`, `zheng2023ccs`, `kwon2023dataoob` | Moderate-DS, CCS, Data-OOB; adapter tabular/OOF không được gọi nguyên bản nếu đã đổi representation |
| Kernel compression | `dwivedi2024kernel`, `shetty2022compress` | Kernel Thinning / Compress++; dùng đúng thuật toán adapter thực gọi, cite cả hai không biến KT thành Compress++ |
| Cơ chế SKQ | `rahimi2007rff`, `chen2010herding`, `gonzalez1985clustering` | RFF, herding, Gonzalez; đây là thành phần kế thừa, không là đóng góp mới |
| Chất lượng khóa học | `abda2024quality` | Online 28/12/2023, volume 29 năm 2024; giữ năm trích dẫn của volume |
| Chất lượng khóa học | `nguyen2027lodibi` | Theo Springer: hội nghị SOICT 2025, online 12/08/2026, năm citation 2027. Không tự đổi ba mốc thành cùng một năm |
| Chất lượng dữ liệu / view | `nguyenTempoDQ` | Publisher xác nhận title, tác giả và DOI. Chờ citation export để bổ sung volume/issue/year; chưa suy năm từ DOI. Repo chính thức: [TEMPO-DQ](https://github.com/datachain-uit/TEMPO-DQ-Framework) |
| Kịch bản mới / đã gặp | `whitehill2017measure`, `whitehill2017delving`, `schein2002cold` | Whitehill short paper và expanded preprint cùng tác giả, không đếm là hai bằng chứng độc lập. Schein thuộc recommendation; không đồng nhất đối tượng mới ở đó với giao thức CQ |
| Shift / prior | `moreno2012shift`, `saerens2002prior` | Nêu giả định label shift khi dùng A8; A8 không tự sửa covariate hoặc concept shift |
| Tách lựa chọn và đánh giá | `cawley2010selection` | Tuning dev, freeze và đánh giá test tách biệt |

## Nguồn chính đã kiểm metadata

- [CoreTab / PVLDB](https://www.vldb.org/pvldb/vol18/p876-razmadze.pdf)
- [BDIS / AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20578)
- [Moderate-DS / repo tác giả](https://github.com/tmllab/2023_ICLR_Moderate-DS)
- [CCS / preprint tác giả](https://arxiv.org/abs/2210.15809)
- [Data-OOB / PMLR](https://proceedings.mlr.press/v202/kwon23e.html)
- [Kernel Thinning / JMLR](https://www.jmlr.org/papers/v25/21-1334.html)
- [Compress++ / preprint tác giả](https://arxiv.org/abs/2111.07941)
- [RFF / NeurIPS](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
- [Herding / UAI, PMLR reissue](https://proceedings.mlr.press/r8/chen10a.html)
- [Gonzalez / publisher](https://www.sciencedirect.com/science/article/pii/0304397585902245)
- [Abda / Springer](https://link.springer.com/article/10.1007/s10639-023-12247-w)
- [LoDiBi / Springer](https://link.springer.com/chapter/10.1007/978-981-92-2587-3_31)
- [TEMPO-DQ / publisher](https://www.sciencedirect.com/science/article/pii/S0306457326005650)
- [Whitehill short / tác giả](https://scholar.harvard.edu/files/dtingley/files/whitehill-mooc.pdf), [expanded preprint](https://arxiv.org/abs/1702.06404)

Đã bổ sung 16 mục vào bốn tài liệu nền, tổng 20 khóa không trùng. Đây là kiểm metadata và phạm vi trích dẫn. Chưa ghi Y17 hoàn tất toàn bộ: cần chốt metadata TEMPO-DQ còn thiếu và kiểm nội dung toàn văn khi viết chương tổng quan; không dùng abstract/metadata để khẳng định tái lập hay so sánh trực tiếp với CQ.
