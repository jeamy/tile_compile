"""
Parallel tile processing for Phase 6 (TILE_RECONSTRUCTION_TLR)
Methodology v4 - Production-ready implementation
"""

from runner.tile_processor_v4 import TileProcessor, TileProcessorConfig


def process_tile_job(args):
    """
    Worker function for parallel tile processing.
    
    Args:
        args: tuple of (tile_id, bbox, frame_paths, global_weights, cfg)
    
    Returns:
        tuple: (tile_id, bbox, tile_img, warps, metadata)
    """
    tile_id, bbox, frame_paths, global_weights, cfg = args
    
    # Create config object
    cfg_obj = TileProcessorConfig(cfg)
    
    tp = TileProcessor(
        tile_id=tile_id,
        bbox=bbox,
        frame_paths=frame_paths,
        global_weights=global_weights,
        cfg=cfg_obj,
    )
    
    tile_img, warps = tp.run()

    meta = tp.get_metadata()
    
    return tile_id, bbox, tile_img, warps, meta
