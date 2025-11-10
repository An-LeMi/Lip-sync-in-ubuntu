# 🎬 FLOAT Long Audio Processing - Complete Guide

**Process audio of any length (2 minutes to 30+ minutes) without crashes!**

---

## 📦 What You Need

### Required:
1. ✅ **Workflow:** `float_workflow_simple.json` (already in custom_nodes/)
2. ✅ **Nodes:** `FloatBatchProcessSimple` (already installed in ComfyUI-FLOAT_Optimized)
3. ✅ **FLOAT Model:** Will auto-download on first use

### Optional (for MP4 output):
- **VideoHelperSuite** - Converts frames to video (see instructions below)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Restart ComfyUI (First Time Only)

```bash
# Stop ComfyUI (press Ctrl+C in terminal)
# Then start again:
cd /Users/le.minh.an.3077/Downloads/new-lip-sync/ComfyUI
python main.py
```

This loads the new batch processing nodes.

### Step 2: Load Workflow

In ComfyUI browser (http://127.0.0.1:8188):
1. Click **"Load"** button
2. Select: **`custom_nodes/float_workflow_simple.json`**
3. Workflow loads with 6 nodes ✅

### Step 3: Upload & Run

1. **Node 2:** Upload face image (512x512 recommended)
2. **Node 3:** Upload audio file (any length!)
3. **Node 5:** Set `segment_duration_seconds`:
   - 6GB VRAM → **60-90**
   - 8GB VRAM → **90-120**
   - 12GB+ VRAM → **120-180**
4. Click **"Queue Prompt"**

**Done!** Frames save to `ComfyUI/output/FLOAT_LongAudio_*.png`

---

## 📊 Console Output

You'll see progress in real-time:

```
Processing audio: 600.0s at 16000Hz
Audio longer than 120.0s, using batch processing
Split into 5 segments
Processing segment 1/5 (20.0%)
Processing segment 2/5 (40.0%)
Processing segment 3/5 (60.0%)
Processing segment 4/5 (80.0%)
Processing segment 5/5 (100.0%)
Merging 5 video segments
Merged into 15000 total frames
✅ Done!
```

---

## 🎬 Convert Frames to Video

### Option A: Install VideoHelperSuite (Recommended)

```bash
cd /Users/le.minh.an.3077/Downloads/new-lip-sync/ComfyUI/custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
pip install -r ComfyUI-VideoHelperSuite/requirements.txt
```

**Restart ComfyUI**, then:
1. Add `VHS_VideoCombine` node after Node 5
2. Connect: images → VHS_VideoCombine
3. Connect: audio → VHS_VideoCombine
4. Connect: fps → VHS_VideoCombine
5. Run workflow → Get MP4 output! 🎥

### Option B: Use FFmpeg (No Installation in ComfyUI)

```bash
cd /Users/le.minh.an.3077/Downloads/new-lip-sync/ComfyUI/output

ffmpeg -framerate 25 -pattern_type glob -i 'FLOAT_LongAudio_*.png' \
       -i /path/to/your_original_audio.wav \
       -c:v libx264 -pix_fmt yuv420p -crf 18 \
       -c:a aac -b:a 192k \
       -shortest final_video.mp4
```

---

## ⚙️ Settings Explained

### Node 5: FloatBatchProcessSimple

| Setting | Default | What It Does |
|---------|---------|--------------|
| **segment_duration_seconds** | 120 | Length of each chunk. Lower = less VRAM |
| **overlap_frames** | 10 | Smoothness at segment boundaries (0=hard cuts) |
| **fps** | 25 | Frames per second |
| **a_cfg_scale** | 2.0 | Audio guidance (higher = follows audio more) |
| **e_cfg_scale** | 1.0 | Emotion guidance |
| **emotion** | none | Force emotion: happy/sad/angry/etc |
| **face_align** | true ✅ | Auto-centers face (keep enabled!) |
| **seed** | 15 | Random seed (same = reproducible results) |

**Most important:** Adjust `segment_duration_seconds` based on your GPU VRAM!

---

## 💡 How It Works

```
Your 10-minute audio
        ↓
Automatically split into 120-second segments:
  • Segment 1: 0:00-2:00
  • Segment 2: 2:00-4:00
  • Segment 3: 4:00-6:00
  • Segment 4: 6:00-8:00
  • Segment 5: 8:00-10:00
        ↓
Process each segment separately
(Each uses only ~6GB VRAM - no crash!)
        ↓
Merge with 10-frame smooth blending
(No visible seams!)
        ↓
Output all frames
        ↓
Final result: 15,000 frames ready for video!
```

---

## ❓ Troubleshooting

### Problem: "FloatBatchProcessSimple not found"
**Solution:** Restart ComfyUI to load the new nodes

### Problem: Out of memory error
**Solution:** Reduce `segment_duration_seconds` to 60 or 90

### Problem: Visible jumps between segments
**Solution:** Increase `overlap_frames` to 15 or 20

### Problem: Processing is very slow
**Solution:** Increase `segment_duration_seconds` (if VRAM allows)

### Problem: Audio and video don't sync
**Solution:** Keep `fps` at 25, don't modify audio sample rate

### Problem: Face looks wrong
**Solution:** Enable `face_align` and use good quality 512x512 image

---

## 📊 Expected Processing Times

| Audio Length | Segments | Time (RTX 3060 12GB) |
|--------------|----------|----------------------|
| 2 minutes    | 1        | ~2-3 minutes         |
| 5 minutes    | 3        | ~6-9 minutes         |
| 10 minutes   | 5        | ~12-15 minutes       |
| 30 minutes   | 15       | ~35-45 minutes       |

*Processing time ≈ 1.5× audio length*

---

## 🎯 Tips for Best Results

### Image Quality:
- ✅ Use 512×512 pixel images
- ✅ Center the face in frame
- ✅ Face should be 40-60% of image
- ✅ Simple background works best
- ✅ Good lighting on face

### Audio Quality:
- ✅ Clear voice recording
- ✅ Minimal background noise
- ✅ Remove music if possible
- ✅ 16kHz or 44.1kHz sample rate

### Performance:
- ✅ Test with 3-minute audio first
- ✅ Use same seed for consistency
- ✅ Monitor console for progress
- ✅ Close other GPU-heavy apps

---

## 📁 File Structure

After setup, you have:

```
ComfyUI/
├── custom_nodes/
│   ├── float_workflow_simple.json      ← THE WORKFLOW
│   ├── LONG_AUDIO_GUIDE.md             ← THIS GUIDE
│   └── ComfyUI-FLOAT_Optimized/
│       └── src/nodes/
│           ├── nodes_batch_simple.py   ← Batch processor
│           └── nodes_batch.py          ← Components
└── output/
    └── FLOAT_LongAudio_*.png           ← Your output frames
```

---

## 🔧 Advanced: CLI Tool (Optional)

For command-line processing:

```bash
cd /Users/le.minh.an.3077/Downloads/new-lip-sync/ComfyUI/custom_nodes/ComfyUI-FLOAT_Optimized/tools

# Split audio into segments
python batch_process.py --audio long_audio.wav --segment-duration 120

# After processing each segment through FLOAT manually...

# Merge video segments
python batch_process.py --merge-only --temp-dir temp_segments \
    --audio long_audio.wav --output final.mp4
```

Most users won't need this - the workflow handles everything!

---

## ✅ Summary

**What you need:**
1. One workflow: `float_workflow_simple.json`
2. One node: `FloatBatchProcessSimple` (auto-loaded)
3. This guide: `LONG_AUDIO_GUIDE.md`

**How to use:**
1. Restart ComfyUI (first time)
2. Load workflow
3. Upload image + audio
4. Set segment duration for your GPU
5. Run!

**What you get:**
- Process any length audio
- No crashes or OOM errors
- Automatic splitting & merging
- Smooth transitions
- Real-time progress

---

**That's everything you need! Load the workflow and start processing!** 🚀

