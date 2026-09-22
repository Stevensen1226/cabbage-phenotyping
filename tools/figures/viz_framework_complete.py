import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_framework():
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Increase figure size again to accommodate larger fonts and spacing
    fig, ax = plt.subplots(figsize=(26, 16)) 
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 13)
    ax.axis('off')

    # Color Palette
    c_input = '#EEEEEE'      
    c_pre = '#D5E8D4'        
    c_pre_dark = '#82B366'
    c_seg = '#DAE8FC'        
    c_seg_dark = '#6C8EBF'
    c_core = '#FFE6CC'       
    c_core_dark = '#D79B00'
    c_trait = '#F8CECC'      
    c_trait_dark = '#B85450'
    
    # FONTS (Significantly LAGER)
    f_title = 23      # Module titles
    f_body = 19       # Normal text
    f_bold = 19       # Bold labels
    
    # Common styles
    # zorder=20 for text to ensure it's on top of everything
    txt_style = {'ha': 'center', 'va': 'center', 'fontsize': f_body, 'zorder': 20}
    module_style = {'fontsize': f_title, 'fontweight': 'bold', 'zorder': 20}
    
    # COORDINATES & SIZES
    # Global Shift
    x_input = 0.5
    
    # Module A
    x_modA = 4.0
    w_modA = 4.0
    h_modA = 7.0 # Increase height
    
    # Gap
    gap_AC = 2.5
    
    # Module C
    x_modC = x_modA + w_modA + gap_AC # 4+4+2.5 = 10.5
    w_modC = 8.0 # Wider for big text
    h_modC = 9.0 
    
    # Module D/E
    gap_CD = 1.0
    x_modDE = x_modC + w_modC + gap_CD # 10.5 + 8 + 1 = 19.5
    w_modDE = 5.5
    
    # ========================
    # 1. INPUT
    # ========================
    # Input box
    # Center y around 10.5
    input_h = 1.5
    input_y = 10.0
    ax.add_patch(patches.FancyBboxPatch((x_input, input_y), 2.8, input_h, boxstyle="round,pad=0.1", fc=c_input, ec='black', zorder=5))
    ax.text(x_input + 1.4, input_y + input_h/2, "Input\nRaw Point Cloud", fontweight='bold', fontsize=f_bold, ha='center', va='center', zorder=20)

    # Arrow Input -> Module A
    # Start: Right of input box
    # End: Left of ModA box
    start_x = x_input + 2.8
    end_x = x_modA
    ax.arrow(start_x, input_y + input_h/2, end_x - start_x, 0, head_width=0.2, color='black', length_includes_head=True, zorder=2)

    # ========================
    # 2. MODULE A: Preprocessing
    # ========================
    # Vertical position: Top aligned with Input roughly
    # Top y = 12.0
    y_modA = 5.0
    ax.add_patch(patches.FancyBboxPatch((x_modA, y_modA), w_modA, h_modA, boxstyle="round,pad=0.1", fc=c_pre, ec=c_pre_dark, lw=3, zorder=1)) # Thicker border
    
    # Title inside box at top
    # Box top is y_modA + h_modA = 12.0
    ax.text(x_modA + w_modA/2, y_modA + h_modA - 0.5, "Module A:\nPreprocessing", color=c_pre_dark, ha='center', va='center', **module_style)
    
    steps_pre = [
        "ROI Cropping", 
        "SOR Denoising", 
        "CSF Ground Filter", 
        "Height Normalization",
        "Semantic Seg\n(PointNet++)"
    ]
    
    # Distribute steps
    # Available height for steps: 12.0 - 0.5 (title) - 5.0 = 6.5 space
    current_y = y_modA + h_modA - 1.5 
    step_h = 0.8
    step_gap = 1.1
    
    for i, step in enumerate(steps_pre):
        # Box
        ax.add_patch(patches.Rectangle((x_modA + 0.3, current_y - step_h/2), w_modA - 0.6, step_h, fc='white', ec=c_pre_dark, zorder=5))
        ax.text(x_modA + w_modA/2, current_y, step, **txt_style)
        
        # Arrow to next
        if i < len(steps_pre) - 1:
            next_y = current_y - step_gap
            ax.arrow(x_modA + w_modA/2, current_y - step_h/2, 0, -(step_gap - step_h), head_width=0.15, fc=c_pre_dark, ec=c_pre_dark, zorder=2, length_includes_head=True)
            current_y = next_y

    # Arrow Module A -> Module B
    # From top-ish of A to top-ish of B/C area
    # Exit point: Right side of A
    # Let's align with the Input arrow y-level roughly or slightly below?
    # Say y=10.5
    ax.arrow(x_modA + w_modA, 10.5, gap_AC, 0, head_width=0.2, color='black', zorder=2, length_includes_head=True)

    # ========================
    # 3. MODULE B: Coarse Segmentation
    # ========================
    # Placed at top of C column
    # Mod C will reside below it.
    w_modB = 4.0 
    h_modB = 2.8
    x_modB = x_modC
    y_modB = 9.2 # Top at 12.0
    
    ax.add_patch(patches.FancyBboxPatch((x_modB, y_modB), w_modB, h_modB, boxstyle="round,pad=0.1", fc=c_seg, ec=c_seg_dark, lw=3, zorder=1))
    ax.text(x_modB + w_modB/2 - 2, y_modB + h_modB - 0.5, "Module B: Coarse Seg", color=c_seg_dark, **module_style)
    
    # Internal Step
    step_B_y = y_modB + 1.2
    ax.add_patch(patches.Rectangle((x_modB + 0.3, step_B_y - 0.4), w_modB - 0.6, 0.8, fc='white', ec=c_seg_dark, zorder=5))
    ax.text(x_modB + w_modB/2, step_B_y, "3D Watershed\nAlgorithm", **txt_style)
    
    # Output
    ax.text(x_modB + w_modB/2, y_modB + 0.5, "Candidate Clusters", fontsize=f_bold, fontweight='bold', color='black', ha='center', va='center', zorder=20)

    # Arrow Module B -> Module C
    ax.arrow(x_modB + w_modB/2, y_modB, 0, -0.4, head_width=0.2, color='black', zorder=2, length_includes_head=True)

    # ========================
    # 4. MODULE C: RSS Refinement
    # ========================
    # Below B
    # Top of C = y_modB - space. Say start C box at y=8.5? arrow is length 0.4. y_modB is 9.2. Gap is 9.2 - 0.4 = 8.8?
    y_modC_top = 8.8 # Actually box top logic
    h_modC = 8.3
    y_modC = 0.5
    
    ax.add_patch(patches.FancyBboxPatch((x_modC, y_modC), w_modC, h_modC, boxstyle="round,pad=0.1", fc=c_core, ec=c_core_dark, lw=4, ls='-', zorder=1)) # Thicker border for Core
    ax.text(x_modC + w_modC/2 - 3.5, y_modC + h_modC - 0.9, "Module C:\nStructure-Aware \nRecursive Skeleton Splitting (SARR-SS)", color=c_core_dark, **module_style)
    
    # Steps inside C
    # Shift slightly right to allow loop arrow on left
    cx_stepC = x_modC + 3.0
    w_stepC = 4.0
    
    # 4.1 Screening
    y_screen = 7.0
    ax.add_patch(patches.Rectangle((cx_stepC - w_stepC/2, y_screen - 0.6), w_stepC, 1.2, fc='#FFF2CC', ec=c_core_dark, lw=2, zorder=5))
    ax.text(cx_stepC, y_screen, "Adhesion Screening\n(Shape/Ratio)", fontsize=f_bold, fontweight='bold', ha='center', va='center', zorder=20)
    
    # Path 1: Pass
    start_pass_x = cx_stepC + w_stepC/2
    end_pass_x = x_modDE
    ax.arrow(start_pass_x, y_screen, end_pass_x - start_pass_x, 0, head_width=0.2, color='#2D7D32', length_includes_head=True, zorder=3, lw=2.5)
    ax.text(start_pass_x + 1.8, y_screen + 0.4, "No Adhesion", color='#2D7D32', fontsize=f_bold, fontweight='bold', ha='center',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=3), zorder=20)
            
    # Path 2: Fail (Down)
    ax.arrow(cx_stepC, y_screen - 0.6, 0, -1.0, head_width=0.2, color='#C00000', zorder=3, lw=2.5, length_includes_head=True)
    ax.text(cx_stepC + 0.3, y_screen - 1.1, "Adhesion\nDetected", color='#C00000', fontsize=14, fontweight='bold', ha='left', va='center', zorder=20)
    
    # 4.2 Skeleton
    y_skel = 4.8
    ax.add_patch(patches.Rectangle((cx_stepC - w_stepC/2, y_skel - 0.5), w_stepC, 1.0, fc='white', ec=c_core_dark, zorder=5))
    ax.text(cx_stepC, y_skel, "Skeleton Extraction\n(PCA Principal Axis)", **txt_style)
    ax.arrow(cx_stepC, y_skel - 0.5, 0, -0.8, head_width=0.15, ec=c_core_dark, fc=c_core_dark, zorder=2, length_includes_head=True)
    
    # 4.3 Density
    y_dens = 3.0
    ax.add_patch(patches.Rectangle((cx_stepC - w_stepC/2, y_dens - 0.5), w_stepC, 1.0, fc='white', ec=c_core_dark, zorder=5))
    ax.text(cx_stepC, y_dens, "1D Density Analysis\n(Peak-Valley Detection)", **txt_style)
    ax.arrow(cx_stepC, y_dens - 0.5, 0, -0.8, head_width=0.15, ec=c_core_dark, fc=c_core_dark, zorder=2, length_includes_head=True)
    
    # 4.4 Splitting
    y_split = 1.2
    ax.add_patch(patches.Rectangle((cx_stepC - w_stepC/2, y_split - 0.5), w_stepC, 1.0, fc='white', ec=c_core_dark, zorder=5))
    ax.text(cx_stepC, y_split, "Geometric Splitting\n(Cutting Plane)", **txt_style)
    
    # RECURSIVE LOOP
    # From left of Splitting to left of Screening
    # Use explicit dashed lines
    x_loop_start = cx_stepC - w_stepC/2
    y_loop_start = y_split
    x_loop_end = cx_stepC - w_stepC/2
    y_loop_end = y_screen
    
    # Use a large rad to make it loop out into the gap
    # Note: patches.ArrowStyle is set in FancyArrowPatch
    style = "Simple, tail_width=0.8, head_width=5, head_length=10"
    # To fix dashed line "connected" look, we use a custom linestyle tuple: (offset, (on_len, off_len))
    # e.g. (0, (10, 10))
    loop_arrow = patches.FancyArrowPatch(
        (x_loop_start, y_loop_start),
        (x_loop_end, y_loop_end),
        connectionstyle="arc3,rad=-1.0", # Very curved to go left
        arrowstyle=style,
        color='#C00000',
        lw=2.5,
        linestyle=(0, (8, 6)), # Custom dash: 8pts line, 6pts space
        zorder=30
    )
    ax.add_patch(loop_arrow)
    
    # Loop Text
    # Calculate approx midpoint manually
    # Apex of arc is to the left.
    ax.text(x_modC - 1.2, (y_loop_start + y_loop_end)/2, "Recursive Refinement Loop", 
            color='#C00000', rotation=90, fontweight='bold', fontsize=f_bold, ha='center', va='center', zorder=30)
            
            
    # ========================
    # 5. MODULE D: Optimization
    # ========================
    y_modD = 6.0
    h_modD = 3.0
    ax.add_patch(patches.FancyBboxPatch((x_modDE, y_modD), w_modDE, h_modD, boxstyle="round,pad=0.1", fc=c_seg, ec=c_seg_dark, lw=3, zorder=1))
    ax.text(x_modDE + w_modDE/2 - 1.1, y_modD + h_modD - 0.6, "Module D:\nOptimization", color=c_seg_dark, **module_style)
    
    ax.add_patch(patches.Rectangle((x_modDE + 0.3, y_modD + 0.8), w_modDE - 0.6, 1.0, fc='white', ec=c_seg_dark, zorder=5))
    ax.text(x_modDE + w_modDE/2, y_modD + 1.3, "Fragment Merging\n(Neighborhood Voting)", **txt_style)
    
    # Arrow to E
    ax.arrow(x_modDE + w_modDE/2, y_modD, 0, -0.5, head_width=0.2, color='black', zorder=2, length_includes_head=True)
    
    # ========================
    # 6. MODULE E: Trait Extraction
    # ========================
    y_modE = 0.5
    h_modE = 5.0
    ax.add_patch(patches.FancyBboxPatch((x_modDE, y_modE), w_modDE, h_modE, boxstyle="round,pad=0.1", fc=c_trait, ec=c_trait_dark, lw=3, zorder=1))
    ax.text(x_modDE + w_modDE/2 - 1.1, y_modE + h_modE - 0.6, "Module E:\nTrait Extraction", color=c_trait_dark, **module_style)
    
    steps_trait = [
        "Height (Z-max)",
        "Diameter (AABB/OBB)",
        "Volume (Convex Hull)",
        "Georeferencing"
    ]
    
    current_y_t = y_modE + h_modE - 1.4
    step_gap_t = 1.0
    for i, step in enumerate(steps_trait):
        ax.add_patch(patches.Rectangle((x_modDE + 0.3, current_y_t - 0.5), w_modDE - 0.6, 1.0, fc='white', ec=c_trait_dark, zorder=5))
        ax.text(x_modDE + w_modDE/2, current_y_t, step, **txt_style)
        
        if i < len(steps_trait)-1:
            ax.arrow(x_modDE + w_modDE/2, current_y_t - 0.5, 0, -(step_gap_t - 0.5), head_width=0.15, fc=c_trait_dark, ec=c_trait_dark, zorder=2, length_includes_head=True)
            current_y_t -= step_gap_t

    output_path = os.path.join(output_dir, 'framework_complete_v4.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Framework complete saved to: {output_path}")

if __name__ == "__main__":
    draw_framework()
