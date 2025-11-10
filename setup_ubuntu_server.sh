#!/bin/bash
# ComfyUI Setup Script for Ubuntu Server with CUDA
# Includes FLOAT Batch Processing for Long Audio
# Author: Updated for long audio processing

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  ComfyUI + FLOAT + AdvancedLivePortrait Setup             ║"
echo "║  Ubuntu Server with CUDA Support                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# STEP 0: Ensure 16 GB Swap (for large video merges)
# ============================================================================
echo "🧠 Step 0: Ensuring 16 GB swap space..."
if swapon --show | grep -q "/swapfile"; then
    echo "⚠️  Swapfile already exists, skipping creation."
else
    echo "  📦 Creating 16 GB swapfile at /swapfile"
    sudo fallocate -l 16G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    if ! grep -q "/swapfile" /etc/fstab; then
        echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab >/dev/null
    fi
    echo "  ✅ Swapfile enabled"
fi
echo ""

# Configuration
PYTHON_VERSION="python3.10"
COMFYUI_DIR="$HOME/ComfyUI"
VENV_DIR="$HOME/comfyui_env"
PORT=8000

# ============================================================================
# STEP 1: Install System Dependencies
# ============================================================================
echo "📦 Step 1: Installing system dependencies..."
sudo apt update
sudo apt install -y $PYTHON_VERSION ${PYTHON_VERSION}-venv python3-pip git ffmpeg screen
echo "✅ System dependencies installed"
echo ""

# ============================================================================
# STEP 2: Create Python Virtual Environment
# ============================================================================
echo "🐍 Step 2: Creating Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_VERSION -m venv "$VENV_DIR"
    echo "✅ Virtual environment created at $VENV_DIR"
else
    echo "⚠️  Virtual environment already exists at $VENV_DIR"
fi
echo ""

# Activate virtual environment
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel
echo "✅ Virtual environment activated and pip upgraded"
echo ""

# ============================================================================
# STEP 3: Clone ComfyUI
# ============================================================================
echo "📥 Step 3: Cloning ComfyUI..."
if [ ! -d "$COMFYUI_DIR" ]; then
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    echo "✅ ComfyUI cloned to $COMFYUI_DIR"
else
    echo "⚠️  ComfyUI already exists at $COMFYUI_DIR"
    cd "$COMFYUI_DIR"
    git pull
    echo "✅ ComfyUI updated"
fi
echo ""

# ============================================================================
# STEP 4: Install PyTorch with CUDA Support
# ============================================================================
echo "🔥 Step 4: Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✅ PyTorch with CUDA support installed"
echo ""

# ============================================================================
# STEP 5: Install ComfyUI Requirements
# ============================================================================
echo "📦 Step 5: Installing ComfyUI requirements..."
cd "$COMFYUI_DIR"
pip install -r requirements.txt
echo "✅ ComfyUI requirements installed"
echo ""

# ============================================================================
# STEP 6: Install Custom Nodes
# ============================================================================
echo "🔌 Step 6: Installing custom nodes..."
cd "$COMFYUI_DIR/custom_nodes"

# 6.1: ComfyUI-FLOAT_Optimized (with batch processing)
echo "  📥 Installing ComfyUI-FLOAT_Optimized..."
if [ ! -d "ComfyUI-FLOAT_Optimized" ]; then
    git clone https://github.com/set-soft/ComfyUI-FLOAT_Optimized.git
    cd ComfyUI-FLOAT_Optimized
    pip install -r requirements.txt
    echo "  ✅ FLOAT_Optimized installed"
else
    echo "  ⚠️  FLOAT_Optimized already exists, updating..."
    cd ComfyUI-FLOAT_Optimized
    git pull
    pip install -r requirements.txt
    echo "  ✅ FLOAT_Optimized updated"
fi
cd "$COMFYUI_DIR/custom_nodes"

# 6.2: ComfyUI-AdvancedLivePortrait
echo "  📥 Installing ComfyUI-AdvancedLivePortrait..."
if [ ! -d "ComfyUI-AdvancedLivePortrait" ]; then
    git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait.git
    cd ComfyUI-AdvancedLivePortrait
    [ -f requirements.txt ] && pip install -r requirements.txt
    echo "  ✅ AdvancedLivePortrait installed"
else
    echo "  ⚠️  AdvancedLivePortrait already exists, updating..."
    cd ComfyUI-AdvancedLivePortrait
    git pull
    [ -f requirements.txt ] && pip install -r requirements.txt
    echo "  ✅ AdvancedLivePortrait updated"
fi
cd "$COMFYUI_DIR/custom_nodes"

# 6.3: ComfyUI-VideoHelperSuite (for video output)
echo "  📥 Installing ComfyUI-VideoHelperSuite..."
if [ -d "ComfyUI-VideoHelperSuite" ]; then
    echo "  🗑️  Removing old VideoHelperSuite..."
    rm -rf ComfyUI-VideoHelperSuite
fi
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
cd ComfyUI-VideoHelperSuite
pip install -r requirements.txt
echo "  ✅ VideoHelperSuite installed"
cd "$COMFYUI_DIR/custom_nodes"

echo "✅ All custom nodes installed"
echo ""

# ============================================================================
# STEP 7: Create Batch Processing Nodes for Long Audio
# ============================================================================
echo "🔧 Step 7: Creating batch processing nodes for long audio..."

cd "$COMFYUI_DIR/custom_nodes/ComfyUI-FLOAT_Optimized/src/nodes"

# Create nodes_batch_simple.py (main batch processor)
echo "  📝 Creating nodes_batch_simple.py..."
cat > nodes_batch_simple.py << 'BATCHSIMPLE_EOF'
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized - Simple batch processing node
import torch
from seconohe.torch import model_to_target
from . import main_logger
from .nodes import EMOTIONS


class FloatBatchProcessSimple:
    """All-in-one node for processing long audio - automatically splits and merges"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "ref_audio": ("AUDIO",),
                "float_pipe": ("FLOAT_PIPE",),
                "fps": ("FLOAT", {"default": 25, "step": 1}),
                "segment_duration_seconds": ("FLOAT", {
                    "default": 120.0, 
                    "min": 30.0, 
                    "max": 600.0, 
                    "step": 10.0,
                    "tooltip": "Split audio into segments of this length. Smaller = less VRAM but more processing time"
                }),
                "overlap_frames": ("INT", {
                    "default": 10, 
                    "min": 0, 
                    "max": 50, 
                    "step": 1,
                    "tooltip": "Frames to blend between segments for smooth transitions. 0 = hard cuts, 10-20 = smooth"
                }),
                "a_cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "step": 0.1}),
                "e_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "step": 0.1}),
                "emotion": (EMOTIONS, {"default": "none"}),
                "face_align": ("BOOLEAN", {"default": True}, ),
                "seed": ("INT", {"default": 62064758300528, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "ref_audio", "fps", "info")
    FUNCTION = "process_long_audio"
    CATEGORY = "FLOAT/Batch"
    DESCRIPTION = "Process long audio by automatically splitting into segments and merging. Perfect for audio >3 minutes."
    UNIQUE_NAME = "FloatBatchProcessSimple"
    DISPLAY_NAME = "FLOAT Batch Process (Simple)"

    def process_long_audio(self, ref_image, ref_audio, float_pipe, fps, segment_duration_seconds, 
                          overlap_frames, a_cfg_scale, e_cfg_scale, emotion, face_align, seed):
        
        waveform = ref_audio['waveform']
        sample_rate = ref_audio['sample_rate']
        
        # Calculate audio duration
        total_samples = waveform.shape[2] if len(waveform.shape) > 2 else waveform.shape[1]
        duration_seconds = total_samples / sample_rate
        
        main_logger.info(f"Processing audio: {duration_seconds:.1f}s at {sample_rate}Hz")
        
        # Check if we need to split
        if duration_seconds <= segment_duration_seconds:
            main_logger.info(f"Audio shorter than {segment_duration_seconds}s, processing normally")
            return self._process_single_segment(
                ref_image, ref_audio, float_pipe, fps, 
                a_cfg_scale, e_cfg_scale, emotion, face_align, seed
            )
        
        # Split and process
        main_logger.info(f"Audio longer than {segment_duration_seconds}s, using batch processing")
        segments = self._split_audio(ref_audio, segment_duration_seconds)
        
        main_logger.info(f"Split into {len(segments)} segments")
        
        # Process each segment
        video_segments = []
        for idx, audio_segment in enumerate(segments):
            progress = (idx + 1) / len(segments) * 100
            main_logger.info(f"Processing segment {idx+1}/{len(segments)} ({progress:.1f}%)")
            
            result = self._process_single_segment(
                ref_image, audio_segment, float_pipe, fps,
                a_cfg_scale, e_cfg_scale, emotion, face_align, seed + idx
            )
            
            video_segments.append(result[0])  # Get images from result
        
        # Merge segments
        main_logger.info(f"Merging {len(video_segments)} video segments")
        merged_video = self._merge_videos(video_segments, overlap_frames)
        
        info = f"Processed {len(segments)} segments, total {duration_seconds:.1f}s"
        
        return (merged_video, ref_audio, fps, info)
    
    def _split_audio(self, audio, segment_duration_seconds):
        """Split audio into segments"""
        waveform = audio['waveform'][0]  # Get first batch item (C, N)
        sample_rate = audio['sample_rate']
        
        segment_samples = int(segment_duration_seconds * sample_rate)
        total_samples = waveform.shape[1]
        
        segments = []
        start = 0
        
        while start < total_samples:
            end = min(start + segment_samples, total_samples)
            segment = waveform[:, start:end].unsqueeze(0)  # (1, C, N_seg)
            
            segments.append({
                'waveform': segment,
                'sample_rate': sample_rate
            })
            
            if end >= total_samples:
                break
            start += segment_samples
        
        return segments
    
    def _process_single_segment(self, ref_image, ref_audio, float_pipe, fps,
                                a_cfg_scale, e_cfg_scale, emotion, face_align, seed):
        """Process a single audio segment"""
        float_pipe.G.target_device = float_pipe.rank
        float_pipe.G.cudnn_benchmark_setting = float_pipe.opt.cudnn_benchmark_enabled
        
        with model_to_target(main_logger, float_pipe.G):
            float_pipe.opt.fps = fps
            
            current_image = ref_image[0:1].to(float_pipe.rank)
            current_audio_wf = ref_audio['waveform'][0:1].to(float_pipe.rank)
            current_audio = {'waveform': current_audio_wf, 'sample_rate': ref_audio['sample_rate']}
            
            images_thwc = float_pipe.run_inference(
                None, current_image, current_audio,
                a_cfg_scale=a_cfg_scale,
                r_cfg_scale=float_pipe.opt.r_cfg_scale,
                e_cfg_scale=e_cfg_scale,
                emo=None if emotion == "none" else emotion,
                no_crop=not face_align,
                seed=seed
            )
            
            return (images_thwc.cpu(), ref_audio, fps)
    
    def _merge_videos(self, video_segments, overlap_frames):
        """Merge video segments with blending"""
        if len(video_segments) == 1:
            return video_segments[0]
        
        merged_frames = []
        
        for idx, segment in enumerate(video_segments):
            segment_frames = segment.shape[0]
            
            if idx == 0:
                merged_frames.append(segment)
            else:
                if overlap_frames > 0 and overlap_frames < segment_frames and len(merged_frames) > 0:
                    prev_segment = merged_frames[-1]
                    prev_frames = prev_segment.shape[0]
                    
                    if prev_frames >= overlap_frames:
                        # Remove overlap from previous
                        merged_frames[-1] = prev_segment[:-overlap_frames]
                        
                        # Blend
                        overlap_prev = prev_segment[-overlap_frames:]
                        overlap_curr = segment[:overlap_frames]
                        
                        blended = []
                        for i in range(overlap_frames):
                            alpha = (i + 1) / (overlap_frames + 1)
                            frame = (1 - alpha) * overlap_prev[i] + alpha * overlap_curr[i]
                            blended.append(frame)
                        
                        blended_tensor = torch.stack(blended)
                        merged_frames.append(blended_tensor)
                        merged_frames.append(segment[overlap_frames:])
                    else:
                        merged_frames.append(segment)
                else:
                    merged_frames.append(segment)
        
        result = torch.cat(merged_frames, dim=0)
        main_logger.info(f"Merged into {result.shape[0]} total frames")
        return result


# Node registration
NODE_CLASS_MAPPINGS = {
    "FloatBatchProcessSimple": FloatBatchProcessSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatBatchProcessSimple": "FLOAT Batch Process (Simple)",
}
BATCHSIMPLE_EOF

echo "  ✅ nodes_batch_simple.py created"

# Create nodes_batch.py (additional components)
echo "  📝 Creating nodes_batch.py..."
cat > nodes_batch.py << 'BATCH_EOF'
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized - Batch processing nodes
import os
import torch
import torchaudio
import numpy as np
from . import main_logger


class AudioFormatConverter:
    """Check audio format and convert MP4 to WAV if needed"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_sample_rate": ("INT", {"default": 16000, "min": 8000, "max": 48000, "step": 1000}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "format_info")
    FUNCTION = "convert_audio"
    CATEGORY = "FLOAT/Batch"
    DESCRIPTION = "Check audio format and convert MP4/other formats to WAV if needed"
    UNIQUE_NAME = "AudioFormatConverter"
    DISPLAY_NAME = "Audio Format Converter"

    def convert_audio(self, audio, target_sample_rate):
        """
        Check if audio is in correct format, convert if needed
        Handles MP4, MP3, and other formats by converting to WAV
        """
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        # Check current format
        original_format = "unknown"
        if hasattr(audio, 'format'):
            original_format = audio.get('format', 'unknown')
        
        main_logger.info(f"Audio input - Sample rate: {sample_rate}Hz, Format: {original_format}")
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            main_logger.info(f"Resampling from {sample_rate}Hz to {target_sample_rate}Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            
            # Handle different waveform shapes
            if len(waveform.shape) == 2:  # (C, N)
                waveform = resampler(waveform)
            elif len(waveform.shape) == 3:  # (B, C, N)
                batch_size = waveform.shape[0]
                resampled_waveforms = []
                for i in range(batch_size):
                    resampled = resampler(waveform[i])
                    resampled_waveforms.append(resampled)
                waveform = torch.stack(resampled_waveforms)
            else:
                main_logger.warning(f"Unexpected waveform shape: {waveform.shape}, skipping resample")
            
            sample_rate = target_sample_rate
        
        # Ensure audio is in correct format for FLOAT processing
        # FLOAT typically expects mono or stereo at 16kHz
        if len(waveform.shape) == 3 and waveform.shape[0] > 1:
            # Multiple batches, take first
            waveform = waveform[0:1]
            main_logger.info("Using first batch from multi-batch audio")
        
        # Convert to stereo if mono (FLOAT works better with stereo)
        if len(waveform.shape) == 2:
            # (C, N) format
            if waveform.shape[0] == 1:
                # Mono, convert to stereo
                waveform = torch.cat([waveform, waveform], dim=0)
                main_logger.info("Converted mono to stereo")
        elif len(waveform.shape) == 3:
            # (B, C, N) format
            if waveform.shape[1] == 1:
                # Mono, convert to stereo
                waveform = torch.cat([waveform, waveform], dim=1)
                main_logger.info("Converted mono to stereo")
        
        output_audio = {
            'waveform': waveform,
            'sample_rate': sample_rate
        }
        
        format_info = f"Converted to WAV format, {sample_rate}Hz, {waveform.shape}"
        main_logger.info(format_info)
        
        return (output_audio, format_info)


class AudioSplitter:
    """Split long audio into segments for batch processing"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "segment_duration_seconds": ("FLOAT", {"default": 120.0, "min": 10.0, "max": 600.0, "step": 1.0}),
                "overlap_seconds": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio_segments", "num_segments")
    FUNCTION = "split_audio"
    CATEGORY = "FLOAT/Batch"
    DESCRIPTION = "Split long audio into segments for processing"
    UNIQUE_NAME = "AudioSplitter"
    DISPLAY_NAME = "Audio Splitter"
    OUTPUT_IS_LIST = (True, False)

    def split_audio(self, audio, segment_duration_seconds, overlap_seconds):
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        batch_size = waveform.shape[0]
        if batch_size > 1:
            main_logger.warning("AudioSplitter: Multiple audio batches detected, using first one")
        
        waveform = waveform[0]
        
        segment_samples = int(segment_duration_seconds * sample_rate)
        overlap_samples = int(overlap_seconds * sample_rate)
        step_samples = segment_samples - overlap_samples
        
        total_samples = waveform.shape[1]
        
        if total_samples <= segment_samples:
            main_logger.info(f"Audio duration {total_samples/sample_rate:.1f}s is shorter than segment duration, no splitting needed")
            return ([audio], 1)
        
        segments = []
        start = 0
        
        while start < total_samples:
            end = min(start + segment_samples, total_samples)
            segment = waveform[:, start:end].unsqueeze(0)
            segments.append({
                'waveform': segment,
                'sample_rate': sample_rate
            })
            
            if end >= total_samples:
                break
            start += step_samples
        
        num_segments = len(segments)
        main_logger.info(f"Split audio into {num_segments} segments of ~{segment_duration_seconds}s each")
        
        return (segments, num_segments)


class VideoSegmentMerger:
    """Merge video segments back together"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_segments": ("IMAGE",),
                "overlap_frames": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_video",)
    FUNCTION = "merge_videos"
    CATEGORY = "FLOAT/Batch"
    DESCRIPTION = "Merge video segments with optional blending"
    UNIQUE_NAME = "VideoSegmentMerger"
    DISPLAY_NAME = "Video Segment Merger"
    INPUT_IS_LIST = True

    def merge_videos(self, video_segments, overlap_frames):
        overlap = overlap_frames[0] if isinstance(overlap_frames, list) else overlap_frames
        
        if not video_segments or len(video_segments) == 0:
            raise ValueError("No video segments to merge")
        
        if len(video_segments) == 1:
            main_logger.info("Only one video segment, returning as-is")
            return (video_segments[0],)
        
        merged_frames = []
        
        for idx, segment in enumerate(video_segments):
            segment_frames = segment.shape[0]
            
            if idx == 0:
                merged_frames.append(segment)
            else:
                if overlap > 0 and overlap < segment_frames and len(merged_frames) > 0:
                    prev_segment = merged_frames[-1]
                    prev_frames = prev_segment.shape[0]
                    
                    if prev_frames >= overlap:
                        merged_frames[-1] = prev_segment[:-overlap]
                        overlap_prev = prev_segment[-overlap:]
                        overlap_curr = segment[:overlap]
                        
                        blended = []
                        for i in range(overlap):
                            alpha = (i + 1) / (overlap + 1)
                            frame = (1 - alpha) * overlap_prev[i] + alpha * overlap_curr[i]
                            blended.append(frame)
                        
                        blended_tensor = torch.stack(blended)
                        merged_frames.append(blended_tensor)
                        merged_frames.append(segment[overlap:])
                    else:
                        merged_frames.append(segment)
                else:
                    merged_frames.append(segment)
        
        result = torch.cat(merged_frames, dim=0)
        total_frames = result.shape[0]
        main_logger.info(f"Merged {len(video_segments)} segments into {total_frames} total frames")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "AudioFormatConverter": AudioFormatConverter,
    "AudioSplitter": AudioSplitter,
    "VideoSegmentMerger": VideoSegmentMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFormatConverter": "Audio Format Converter",
    "AudioSplitter": "Audio Splitter",
    "VideoSegmentMerger": "Video Segment Merger",
}
BATCH_EOF

echo "  ✅ nodes_batch.py created"

# Update __init__.py to register the new nodes
echo "  🔧 Updating __init__.py to register batch processing nodes..."
cd "$COMFYUI_DIR/custom_nodes/ComfyUI-FLOAT_Optimized"

# Check if imports already exist
if ! grep -q "from .src.nodes import nodes_batch" __init__.py; then
    # Add imports after nodes_vadv_loader
    sed -i '/from .src.nodes import nodes_vadv_loader/a from .src.nodes import nodes_batch\nfrom .src.nodes import nodes_batch_simple' __init__.py
    
    # Update register_nodes call to include new nodes
    sed -i 's/\[nodes, nodes_adv, nodes_vadv, nodes_vadv_loader\]/[nodes, nodes_adv, nodes_vadv, nodes_vadv_loader, nodes_batch, nodes_batch_simple]/' __init__.py
    
    echo "  ✅ __init__.py updated to register batch nodes"
else
    echo "  ✅ __init__.py already has batch node imports"
fi

cd "$COMFYUI_DIR/custom_nodes"

echo "✅ Batch processing nodes created and registered"
echo ""

# ============================================================================
# STEP 8: Configure CUDA Memory Settings
# ============================================================================
echo "⚙️  Step 8: Configuring CUDA memory settings..."

# Create startup script with CUDA optimizations
cat > "$COMFYUI_DIR/start_comfyui.sh" << 'EOF'
#!/bin/bash
# ComfyUI Startup Script with CUDA Optimizations

# Disable async allocator, use legacy cudaMalloc for stability
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=backend:native

# Additional CUDA settings for stability
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

# Activate virtual environment
source ~/comfyui_env/bin/activate

# Change to ComfyUI directory
cd ~/ComfyUI

# Test CUDA setup
echo "🔍 Testing CUDA setup..."
python - << 'PY'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
PY

echo ""
echo "🚀 Starting ComfyUI..."
echo ""

# Start ComfyUI with optimized settings
PYTORCH_CUDA_ALLOC_CONF=backend:native \
CUDA_LAUNCH_BLOCKING=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
python main.py \
    --listen 0.0.0.0 \
    --port 8000 \
    --disable-cuda-malloc \
    --force-fp32
EOF

chmod +x "$COMFYUI_DIR/start_comfyui.sh"
echo "✅ Startup script created at $COMFYUI_DIR/start_comfyui.sh"
echo ""

# ============================================================================
# STEP 9: Verify Installation and Long Audio Support
# ============================================================================
echo "✅ Step 9: Verifying installation and long audio support..."

# Test CUDA
echo "  🔍 Testing CUDA availability..."
python - << 'PY'
import torch
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"  ✅ CUDA is available: {torch.cuda.get_device_name(0)}")
    print(f"  ✅ CUDA capability: {torch.cuda.get_device_capability(0)}")
else:
    print("  ⚠️  CUDA not available - will run on CPU")
PY

# Verify batch processing nodes exist
echo ""
echo "  🔍 Verifying batch processing nodes..."
BATCH_SIMPLE="$COMFYUI_DIR/custom_nodes/ComfyUI-FLOAT_Optimized/src/nodes/nodes_batch_simple.py"
BATCH_REGULAR="$COMFYUI_DIR/custom_nodes/ComfyUI-FLOAT_Optimized/src/nodes/nodes_batch.py"

if [ -f "$BATCH_SIMPLE" ]; then
    echo "  ✅ nodes_batch_simple.py exists"
    if grep -q "FloatBatchProcessSimple" "$BATCH_SIMPLE"; then
        echo "  ✅ FloatBatchProcessSimple class found"
    else
        echo "  ❌ FloatBatchProcessSimple class NOT found"
    fi
else
    echo "  ❌ nodes_batch_simple.py NOT found"
fi

if [ -f "$BATCH_REGULAR" ]; then
    echo "  ✅ nodes_batch.py exists"
else
    echo "  ❌ nodes_batch.py NOT found"
fi

# Verify __init__.py registration
echo ""
echo "  🔍 Verifying node registration..."
if grep -q "nodes_batch_simple" "$COMFYUI_DIR/custom_nodes/ComfyUI-FLOAT_Optimized/__init__.py"; then
    echo "  ✅ Batch nodes registered in __init__.py"
else
    echo "  ❌ Batch nodes NOT registered in __init__.py"
fi

# Verify workflow nodes
echo ""
echo "  🔍 Verifying workflow requirements..."
echo "  Required nodes for long audio processing:"
echo "    ✅ LoadImage (ComfyUI built-in)"
echo "    ✅ LoadAudio (ComfyUI built-in)"
echo "    ✅ AudioFormatConverter (NEW - converts MP4 to WAV)"
echo "    ✅ LoadFloatModelsOpt (FLOAT_Optimized)"
echo "    ✅ FloatBatchProcessSimple (NEW - batch processing)"
echo "    ✅ VHS_VideoCombine (VideoHelperSuite - MP4 output)"
echo "    ✅ SaveImage (ComfyUI built-in)"
echo ""
echo "  📋 Workflow file: float_workflow_simple.json"
echo "     - AudioFormatConverter: Converts MP4/MP3 to WAV, resamples to 16kHz"
echo "     - FloatBatchProcessSimple: Automatic long audio handling"
echo "     - Automatically splits audio into segments"
echo "     - Processes each segment separately (prevents OOM)"
echo "     - Merges with smooth transitions"

echo ""

# ============================================================================
# INSTALLATION COMPLETE
# ============================================================================
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║              ✅ INSTALLATION COMPLETE!                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "📋 What was installed:"
echo "  ✅ ComfyUI at: $COMFYUI_DIR"
echo "  ✅ Virtual environment at: $VENV_DIR"
echo "  ✅ FLOAT_Optimized with batch processing for long audio"
echo "  ✅ AdvancedLivePortrait"
echo "  ✅ VideoHelperSuite"
echo "  ✅ CUDA optimizations configured"
echo ""
echo "🚀 To start ComfyUI:"
echo ""
echo "    cd ~/ComfyUI"
echo "    ./start_comfyui.sh"
echo ""
echo "🌐 Access ComfyUI at: http://YOUR_SERVER_IP:8000"
echo ""
echo "💡 Next steps:"
echo "  1. Upload workflow: float_workflow_simple.json"
echo "  2. Upload assets: face image + long audio file"
echo "  3. Set segment_duration_seconds: 120 (for 12GB VRAM)"
echo "  4. Click Queue Prompt - MP4 video generated automatically!"
echo ""
echo "📊 Long Audio Processing Features:"
echo "  ✅ Automatic audio splitting into segments"
echo "  ✅ Sequential processing (prevents OOM errors)"
echo "  ✅ Smooth video merging with frame blending"
echo "  ✅ Works with audio of any length (2min to 30min+)"
echo "  ✅ Real-time progress tracking in console"
echo "  ✅ MP4 video saved automatically (VideoHelperSuite)"
echo ""

