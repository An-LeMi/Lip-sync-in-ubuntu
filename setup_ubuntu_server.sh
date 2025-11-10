#!/bin/bash
# ComfyUI Setup Script for Ubuntu Server with CUDA
# Includes FLOAT Batch Processing for Long Audio
# Author: Updated for long audio processing
# Date: 2025-11-10

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  ComfyUI + FLOAT + AdvancedLivePortrait Setup            ║"
echo "║  Ubuntu Server with CUDA Support                         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
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
# STEP 7: Apply FLOAT Batch Processing Fixes
# ============================================================================
echo "🔧 Step 7: Applying batch processing fixes for long audio..."

# Fix import paths in batch processing nodes (if they haven't been fixed yet)
BATCH_SIMPLE="$COMFYUI_DIR/custom_nodes/ComfyUI-FLOAT_Optimized/src/nodes/nodes_batch_simple.py"
BATCH_REGULAR="$COMFYUI_DIR/custom_nodes/ComfyUI-FLOAT_Optimized/src/nodes/nodes_batch.py"

if [ -f "$BATCH_SIMPLE" ]; then
    if grep -q "from .. import main_logger" "$BATCH_SIMPLE"; then
        echo "  🔧 Fixing imports in nodes_batch_simple.py..."
        sed -i 's/from .. import main_logger/from . import main_logger/g' "$BATCH_SIMPLE"
        echo "  ✅ nodes_batch_simple.py fixed"
    else
        echo "  ✅ nodes_batch_simple.py already fixed"
    fi
fi

if [ -f "$BATCH_REGULAR" ]; then
    if grep -q "from .. import main_logger" "$BATCH_REGULAR"; then
        echo "  🔧 Fixing imports in nodes_batch.py..."
        sed -i 's/from .. import main_logger/from . import main_logger/g' "$BATCH_REGULAR"
        echo "  ✅ nodes_batch.py fixed"
    else
        echo "  ✅ nodes_batch.py already fixed"
    fi
fi

echo "✅ Batch processing fixes applied"
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
# STEP 9: Verify Installation
# ============================================================================
echo "✅ Step 9: Verifying installation..."

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
echo "  1. Upload workflow and assets via web interface"
echo "  2. For long audio: use FloatBatchProcessSimple node"
echo "  3. VHS_VideoCombine: format=video/h264-mp4, pix_fmt=yuv420p"
echo ""

