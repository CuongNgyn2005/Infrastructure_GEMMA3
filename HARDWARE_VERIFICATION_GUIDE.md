# ✅ Hardware Verification Checklist - Board ZCU104 Mới

## 🔧 **TRƯỚC TIÊN - Chạy verification script:**

```bash
cd ~/soc/GEMMA3.cpp-MODEL-IN-FPGA
sudo bash verify_hardware.sh
```

Script sẽ kiểm tra:
- ✅ Phiên bản Linux
- ✅ Tổng dung lượng RAM (phải 2GB)
- ✅ DDR memory layout
- ✅ CMA region location
- ✅ Bitstream status
- ✅ FPGA control access
- ✅ Network availability

---

## 📋 **CHECKLIST CỤ THỂ - Trên board mới:**

### **1️⃣ Check RAM size**
```bash
cat /proc/meminfo | head -1
# Output: MemTotal:        2036116 kB (phải ~2GB)
```

**Nếu RAM ≠ 2GB** → Báo cáo, board có thể khác

---

### **2️⃣ Check DDR layout**
```bash
bash check_fpga_memory.sh
```

**Kết quả PHẢI có:**
```
✓ CMA region confirmed @ 0x6B800000-0x77800000
✓ FPGA buffers @ 0x77C00000 are AFTER CMA
✓ Addresses should be SAFE ✅
```

**Nếu khác** → CẦN điều chỉnh buffer addresses

---

### **3️⃣ Check bitstream đã load chưa**
```bash
sudo fpgautil -d
```

**Output phải show:**
```
Xclbin(s) Loaded
  Shell Name        : ...
  PL Programming    : Enabled
  State             : Loaded
```

**Nếu KHÔNG load:**
```bash
sudo fpgautil -b DATN1_wrapper.bit
sudo fpgautil -d  # Verify lại
```

---

### **4️⃣ Check /dev/mem accessible**
```bash
ls -la /dev/mem
# Output: crw------- 1 root root ...
```

**Nếu không có:**
```bash
sudo modprobe uio_pdrv_genirq
```

---

### **5️⃣ Check network (nếu cần download)**
```bash
ping -c 1 8.8.8.8
```

Only needed nếu cần download bitstream

---

## ⚠️ **Nếu phát hiện KHÁC BIỆT:**

### **Scenario 1: RAM ≠ 2GB**
```
⚠️ Board này có RAM khác
→ Báo kích thước RAM cho tôi
→ Có thể cần điều chỉnh buffer addresses
```

### **Scenario 2: CMA region khác**
```
⚠️ CMA at: 0x12345678-0x22345678 (khác 0x6B800000)
→ Báo cáo output của check_fpga_memory.sh
→ Tôi sẽ tính addresses mới
```

### **Scenario 3: Bitstream chưa load**
```
✓ Load bitstream:
  $ sudo fpgautil -b DATN1_wrapper.bit
✓ Verify:
  $ sudo fpgautil -d
```

### **Scenario 4: /dev/mem không có**
```
✓ Tạo device node:
  $ sudo modprobe uio_pdrv_genirq
✓ Rebuild uio module nếu cần
```

---

## 📊 **Summary - Điều kiện để BUILD:**

**✅ CÓ THỂ BUILD nếu:**
- [ ] RAM = 2GB
- [ ] CMA @ 0x6B800000-0x77800000
- [ ] Bitstream loaded
- [ ] /dev/mem accessible
- [ ] Memory addresses verified SAFE

**❌ KHÔNG BUILD nếu:**
- [ ] RAM ≠ 2GB (hardware khác)
- [ ] CMA khác location (cần điều chỉnh)
- [ ] Bitstream chưa load
- [ ] /dev/mem không accessible

---

## 🎯 **Action Plan:**

### **Trên board ZCU104 mới:**

```bash
# 1. Run hardware verification (5 min)
sudo bash verify_hardware.sh

# 2. Run memory check (1 min)
bash check_fpga_memory.sh

# 3. Check bitstream
sudo fpgautil -d

# 4. Load bitstream if needed
sudo fpgautil -b DATN1_wrapper.bit

# 5. Verify kết quả
bash check_fpga_memory.sh
```

---

## 📞 **Báo cáo kết quả cho tôi:**

```
Báo cáo bao gồm:
[ ] Output của verify_hardware.sh
[ ] Output của check_fpga_memory.sh
[ ] Result của fpgautil -d
[ ] Bất kỳ lỗi nào gặp phải
[ ] Các điểm khác biệt so với board cũ
```

---

## ✅ **Sau khi verified hardware:**

Nếu mọi thứ OK → Có thể tiếp tục BUILD

Nếu có vấn đề → Báo cáo, tôi sẽ điều chỉnh config

---

**Đây là quy trình hợp lý để đảm bảo hardware chuẩn bị sẵn sàng!** ✅
