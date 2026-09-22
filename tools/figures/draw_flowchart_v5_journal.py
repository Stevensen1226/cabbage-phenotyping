
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects

# ==========================================
# Journal-Grade Configuration
# ==========================================
# Font: Arial or similar sans-serif is standard for IEEE/Elsevier
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

STYLE = {
    'linewidth': 1.5,
    'arrow_head_width': 3,
    'arrow_head_length': 4,
    'corner_radius': 2,
    'shadow_offset': (1.0, -1.0),
    'colors': {
        'input':   ('#F8F9FA', '#495057'), # Light Gray / Dark Gray
        'preproc': ('#E3F2FD', '#1565C0'), # Blue 50 / Blue 800
        'coarse':  ('#E0F2F1', '#00695C'), # Teal 50 / Teal 800
        'refine':  ('#FFF3E0', '#E65100'), # Orange 50 / Orange 900
        'decision':('#FFF8E1', '#F57F17'), # Amber 50 / Amber 900
        'ops':     ('#FFEBEE', '#C62828'), # Red 50 / Red 800
        'post':    ('#F3E5F5', '#6A1B9A'), # Purple 50 / Purple 800
        'output':  ('#F1F8E9', '#33691E'), # Light Green / Dark Green
        'text':    '#212121',
        'subtext': '#424242'
    }
}

def draw_process_node(ax, x, y, w, h, title, details=None, style_key='preproc', zorder=20):
    """Draw a standard process node with title and optional details."""
    fill, stroke = STYLE['colors'][style_key]
    
    # 1. Shadow
    shadow = patches.FancyBboxPatch(
        (x + STYLE['shadow_offset'][0], y + STYLE['shadow_offset'][1]), w, h,
        boxstyle=f"round,pad=0,rounding_size={STYLE['corner_radius']}",
        ec="none", fc="black", alpha=0.15, zorder=zorder-5
    )
    ax.add_patch(shadow)

    # 2. Box
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={STYLE['corner_radius']}",
        linewidth=STYLE['linewidth'], edgecolor=stroke, facecolor=fill, zorder=zorder
    )
    ax.add_patch(box)

    # 3. Text
    cx, cy = x + w/2, y + h/2
    
    if details:
        # Title at top
        ax.text(cx, y + h - h*0.25 + 1, title, ha='center', va='center', 
                fontsize=15, fontweight='bold', color=STYLE['colors']['text'], zorder=zorder+1)
        # Separator
        ax.plot([x+w*0.1, x+w*0.9], [cy+2, cy+2], color=stroke, lw=0.5, alpha=0.5, zorder=zorder+1)
        # Details
        ax.text(cx, y + h*0.28 + 1.3, details, ha='center', va='center', 
                fontsize=12, color=STYLE['colors']['subtext'], linespacing=1.3, zorder=zorder+1)
    else:
        # Centered Title
        ax.text(cx, cy, title, ha='center', va='center', 
                fontsize=10, fontweight='bold', color=STYLE['colors']['text'], zorder=zorder+1)
        
    return {
        'N': (cx, y+h), 'S': (cx, y), 'E': (x+w, cy), 'W': (x, cy),
        'NE': (x+w, y+h), 'NW': (x, y+h), 'SE': (x+w, y), 'SW': (x, y)
    }

def draw_decision_node(ax, cx, cy, w, h, text, style_key='decision', zorder=20):
    """Draw a diamond decision node."""
    fill, stroke = STYLE['colors'][style_key]
    
    # Vertices
    verts = [
        (cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy), (cx, cy + h/2)
    ]
    
    # Shadow
    s_verts = [(vx+STYLE['shadow_offset'][0], vy+STYLE['shadow_offset'][1]) for vx, vy in verts]
    shadow_path = Path(s_verts, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    shadow = patches.PathPatch(shadow_path, fc='black', ec='none', alpha=0.15, zorder=zorder-5)
    ax.add_patch(shadow)
    
    # Shape
    path = Path(verts, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    patch = patches.PathPatch(path, fc=fill, ec=stroke, lw=STYLE['linewidth'], zorder=zorder)
    ax.add_patch(patch)
    
    # Text
    ax.text(cx + 0.2, cy, text, ha='center', va='center', fontsize=14.8, fontweight='bold', zorder=zorder+1)
    
    return {
        'N': (cx, cy+h/2), 'S': (cx, cy-h/2), 'E': (cx+w/2, cy), 'W': (cx-w/2, cy)
    }

def draw_connector(ax, p1, p2, type='straight', text=None, text_pos='mid', color='#555555', zorder=10):
    """
    Draw logical connectors. 
    types: 'straight', 'elbow_h' (horiz first), 'elbow_v' (vert first)
    """
    path_codes = [Path.MOVETO]
    path_verts = [p1]
    
    if type == 'straight':
        path_verts.append(p2)
        path_codes.append(Path.LINETO)
    elif type == 'elbow_h': # Horizontal then Vertical
        mid = (p2[0], p1[1])
        path_verts.extend([mid, p2])
        path_codes.extend([Path.LINETO, Path.LINETO])
    elif type == 'elbow_v': # Vertical then Horizontal
        mid = (p1[0], p2[1])
        path_verts.extend([mid, p2])
        path_codes.extend([Path.LINETO, Path.LINETO])
        
    # Draw Line
    path = Path(path_verts, path_codes)
    patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=1.5, zorder=zorder)
    ax.add_patch(patch)
    
    # Draw Arrowhead manually for precision
    end = path_verts[-1]
    start_last = path_verts[-2]
    import numpy as np
    dx = end[0] - start_last[0]
    dy = end[1] - start_last[1]
    length = np.sqrt(dx*dx + dy*dy)
    if length > 0.001:
        dx /= length
        dy /= length
        
        # Arrow shape size
        aw = 2.5
        al = 3.5
        
        # Tip is at 'end'
        # Base center is at end - al*(dx,dy)
        base = (end[0] - al*dx, end[1] - al*dy)
        left = (base[0] - aw*(-dy), base[1] - aw*(dx))
        right = (base[0] + aw*(-dy), base[1] + aw*(dx))
        
        arrow_verts = [end, left, right, end]
        arrow_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        arrow_patch = patches.PathPatch(Path(arrow_verts, arrow_codes), fc=color, ec='none', zorder=zorder)
        ax.add_patch(arrow_patch)
        
    # Label
    if text:
        if text_pos == 'mid':
            if len(path_verts) == 3:
                tx, ty = (p2[0]-5 if p2[0]>p1[0] else p2[0]+5, p1[1]+2)
            else:
                tx, ty = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 2
        
        t = ax.text(tx + 2, ty, text, ha='center', va='center', fontsize=12, color=color, fontweight='bold', zorder=zorder+5)
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

def main():
    # A4 Landscape ratio-ish: 29.7cm x 21cm ~ 12x8.5 inches
    fig, ax = plt.subplots(figsize=(24, 8), dpi=300)

    # Coordinate system: 0-210 (Width), 0-70 (Height)
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 70)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # -------------------------------------------------------------
    # 1. PREPROCESSING (Left Column)
    # -------------------------------------------------------------
    x_col1 = 10
    
    # Input
    n_input = draw_process_node(ax, x_col1, 55, 18, 10, "Input", "Point Cloud\n(.ply)", 'input')
    
    # Preproc
    n_pre = draw_process_node(ax, x_col1, 35, 18, 14, "Preprocessing", "\nROI Crop\nSOR Denoise\nCSF Ground\nNorm DTM", 'preproc')
    
    # Arrow Input -> Pre
    draw_connector(ax, n_input['S'], n_pre['N'], 'straight')
    
    # -------------------------------------------------------------
    # 2. COARSE SEGMENTATION
    # -------------------------------------------------------------
    x_col2 = 40
    
    n_ws = draw_process_node(ax, x_col2, 35, 20, 14, "Coarse Seg", "3D Watershed\nEuclidean Dist\nGaussian Smooth", 'coarse')
    
    # Arrow Pre -> WS
    draw_connector(ax, n_pre['E'], n_ws['W'], 'straight')
    
    # -------------------------------------------------------------
    # 3. RECURSIVE REFINEMENT MODULE (Center Stage)
    # -------------------------------------------------------------
    # Background Group
    mod_x, mod_y = 70, 5
    mod_w, mod_h = 65, 60
    
    # Group Frame
    frame = patches.FancyBboxPatch(
        (mod_x, mod_y), mod_w, mod_h, boxstyle="round,pad=0,rounding_size=3",
        linewidth=1.5, edgecolor='#E65100', facecolor='#FFFBEB', linestyle='--', zorder=5
    )
    ax.add_patch(frame)
    ax.text(mod_x + mod_w/2, mod_y + mod_h - 55, "Structure-Aware Recursive Skeleton Splitting", 
            ha='center', fontsize=15, fontweight='bold', color='#E65100', zorder=10)
    
    # 3.1 Candidate Check (Diamond)
    # Entry point
    cx_d1, cy_d1 = mod_x + 15, 50 
    n_dec1 = draw_decision_node(ax, cx_d1, cy_d1, 20, 15, "Length $L$\n$> 0.45m$", 'decision')
    
    # Arrow WS -> Dec1 (Enter Module)
    draw_connector(ax, n_ws['E'], n_dec1['W'], 'elbow_h')
    
    # 3.2 Skeleton Analysis (Box)
    # Below Dec1/Right
    n_skel = draw_process_node(ax, mod_x + 35, 38, 22, 14, "Analysis", "PCA Axis $v_1$\nDensity Profile $\\rho(s)$", 'ops')
    
    # Arrow Dec1 (Yes) -> Skel
    # Changed to elbow_v to avoid overlapping with Analysis box left edge
    draw_connector(ax, n_dec1['E'], n_skel['W'], 'elbow_v')
    # Manual text placement
    ax.text((n_dec1['E'][0] + n_skel['W'][0])/2 - 1, n_skel['W'][1] + 1.5, "Yes", 
            fontsize=12, fontweight='bold', color='#555555', ha='center', zorder=21)
    
    # 3.3 Adaptive Threshold (Diamond)
    # Below Skel
    cx_d2, cy_d2 = mod_x + 46, 25 
    n_dec2 = draw_decision_node(ax, cx_d2, cy_d2, 20, 15, "Valley Ratio\n$< T_{adapt}$", 'decision')
    
    # Arrow Skel -> Dec2
    draw_connector(ax, n_skel['S'], n_dec2['N'], 'straight')
    
    # 3.4 Cut & Recurse (Process)
    # Left of Dec2
    n_cut = draw_process_node(ax, mod_x + 5, 20, 20, 10, "Split", "Plane $\pi \perp v_1$", 'ops')
    
    # Arrow Dec2 (Yes) -> Cut
    draw_connector(ax, n_dec2['W'], n_cut['E'], 'straight', text="Yes")
    
    # Loop Back: Cut -> Candidate (Dec1)
    # Cut Up -> Dec1 Bottom
    draw_connector(ax, n_cut['N'], n_dec1['S'], 'straight', text="Recurse")
    
    # -------------------------------------------------------------
    # 4. OUT (Post Proc)
    # -------------------------------------------------------------
    x_col4 = 155
    
    # Increased size for text safety
    n_post = draw_process_node(ax, x_col4, 34, 35, 16, "Post-Processing", "k-NN Merge\nFragment Filter\nVoting", 'post')
    
    # Routing Exits from Module
    # 1. From Dec1 (No) -> Post
    # Dec1 Top -> Up -> Right -> Post Top
    p_dec1_no_1 = (n_dec1['N'][0], n_dec1['N'][1] + 3)
    p_dec1_no_2 = (n_post['N'][0], p_dec1_no_1[1]) # Above Post
    
    # Draw polyline for NO
    pts_no1 = [n_dec1['N'], p_dec1_no_1, p_dec1_no_2, n_post['N']]
    path_no1 = Path(pts_no1, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax.add_patch(patches.PathPatch(path_no1, fc='none', ec='#555555', lw=1.5, zorder=5))
    # Arrow at end
    draw_connector(ax, p_dec1_no_2, n_post['N'], 'straight') # Tip
    
    t1 = ax.text(n_dec1['N'][0]+2, n_dec1['N'][1]+3, "No", fontsize=12, fontweight='bold', color='#555555', zorder=21)
    t1.set_bbox(dict(fc='white', alpha=0.9, pad=0.3, ec='none'))

    # 2. From Dec2 (No) -> Post
    # Dec2 E -> Right -> Post W (Left side)
    
    pts_no2_new = [
        n_dec2['E'],
        (130, n_dec2['E'][1]),
        (130, n_post['W'][1]),
        n_post['W']
    ]
    path_no2_new = Path(pts_no2_new, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
    ax.add_patch(patches.PathPatch(path_no2_new, fc='none', ec='#555555', lw=1.5, zorder=5))
    # Arrow tip
    draw_connector(ax, (130, n_post['W'][1]), n_post['W'], 'straight')
    
    t2 = ax.text(n_dec2['E'][0]+3, n_dec2['E'][1]+1, "No", fontsize=12, fontweight='bold', color='#555555', zorder=21)
    t2.set_bbox(dict(fc='white', alpha=0.9, pad=0.3, ec='none'))
    
    # 4. Final Output
    # Renamed to "Output", increased size
    n_out = draw_process_node(ax, x_col4, 12, 35, 14, "Output", "Traits:\nH, W, Vol", 'output')

    # Post to Out
    # Post S -> Out N
    draw_connector(ax, n_post['S'], n_out['N'], 'straight')

    # plt.tight_layout() # Removed to prevent aggressive clipping
    # Increase pad_inches to ensure nothing is cut off
    plt.savefig('paper_figure_1_flowchart_v5_journal.png', bbox_inches='tight', pad_inches=0.2)

if __name__ == "__main__":
    main()
