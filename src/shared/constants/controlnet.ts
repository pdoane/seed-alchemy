// ControlNet constants

import type { Architecture } from "../types/models.js";
import type { ControlNetType, PreprocessorInfo } from "../types/controlnet.js";

// ControlNet type display names
export const CONTROLNET_TYPE_LABELS: Record<ControlNetType, string> = {
  canny: "Canny",
  openpose: "Openpose",
  depth: "Depth",
  normal: "Normal",
  lineart: "Lineart",
  lineart_anime: "Lineart Anime",
  scribble: "Scribble",
  softedge: "Soft Edge",
  segment: "Segment",
  tile: "Tile",
  mlsd: "MLSD",
  shuffle: "Shuffle",
  inpaint: "Inpaint",
  repaint: "Repaint",
};

// ControlNet types supported by each architecture
export const CONTROLNET_TYPES_BY_ARCH: Partial<
  Record<Architecture, ControlNetType[]>
> = {
  sdxl: [
    "openpose",
    "depth",
    "normal",
    "lineart",
    "scribble",
    "segment",
    "tile",
    "repaint",
  ],
  sd15: [
    "canny",
    "openpose",
    "depth",
    "normal",
    "lineart",
    "lineart_anime",
    "scribble",
    "softedge",
    "segment",
    "tile",
    "mlsd",
    "shuffle",
    "inpaint",
  ],
};

// All preprocessors with their category and type mappings
export const PREPROCESSORS: PreprocessorInfo[] = [
  // Pose
  {
    name: "DWPreprocessor",
    category: "Pose",
    types: { sd15: "openpose", sdxl: "openpose" },
  },
  {
    name: "OpenposePreprocessor",
    category: "Pose",
    types: { sd15: "openpose", sdxl: "openpose" },
  },
  {
    name: "DensePosePreprocessor",
    category: "Pose",
    types: { sd15: "openpose", sdxl: "openpose" },
  },
  {
    name: "AnimalPosePreprocessor",
    category: "Pose",
    types: { sd15: "openpose", sdxl: "openpose" },
  },
  // Depth
  {
    name: "DepthAnythingV2Preprocessor",
    category: "Depth",
    types: { sd15: "depth", sdxl: "depth" },
  },
  {
    name: "Zoe-DepthMapPreprocessor",
    category: "Depth",
    types: { sd15: "depth", sdxl: "depth" },
  },
  {
    name: "MiDaS-DepthMapPreprocessor",
    category: "Depth",
    types: { sd15: "depth", sdxl: "depth" },
  },
  {
    name: "Metric3D-DepthMapPreprocessor",
    category: "Depth",
    types: { sd15: "depth", sdxl: "depth" },
  },
  {
    name: "DepthAnythingPreprocessor",
    category: "Depth",
    types: { sd15: "depth", sdxl: "depth" },
  },
  {
    name: "LeReS-DepthMapPreprocessor",
    category: "Depth",
    types: { sd15: "depth", sdxl: "depth" },
  },
  {
    name: "MeshGraphormer-DepthMapPreprocessor",
    category: "Depth",
    types: { sd15: "depth", sdxl: "depth" },
  },
  // Normal
  {
    name: "BAE-NormalMapPreprocessor",
    category: "Normal",
    types: { sd15: "normal", sdxl: "normal" },
  },
  {
    name: "MiDaS-NormalMapPreprocessor",
    category: "Normal",
    types: { sd15: "normal", sdxl: "normal" },
  },
  {
    name: "DSINE-NormalMapPreprocessor",
    category: "Normal",
    types: { sd15: "normal", sdxl: "normal" },
  },
  {
    name: "Metric3D-NormalMapPreprocessor",
    category: "Normal",
    types: { sd15: "normal", sdxl: "normal" },
  },
  // Edge - Canny
  {
    name: "CannyEdgePreprocessor",
    category: "Edge",
    types: { sd15: "canny", sdxl: "lineart" },
  },
  {
    name: "PyraCannyPreprocessor",
    category: "Edge",
    types: { sd15: "canny", sdxl: "lineart" },
  },
  // Edge - Lineart
  {
    name: "LineArtPreprocessor",
    category: "Edge",
    types: { sd15: "lineart", sdxl: "lineart" },
  },
  {
    name: "LineartStandardPreprocessor",
    category: "Edge",
    types: { sd15: "lineart", sdxl: "lineart" },
  },
  {
    name: "AnyLineArtPreprocessor_aux",
    category: "Edge",
    types: { sd15: "lineart", sdxl: "lineart" },
  },
  // Edge - Anime Lineart
  {
    name: "AnimeLineArtPreprocessor",
    category: "Edge",
    types: { sd15: "lineart_anime", sdxl: "lineart" },
  },
  {
    name: "Manga2Anime_LineArt_Preprocessor",
    category: "Edge",
    types: { sd15: "lineart_anime", sdxl: "lineart" },
  },
  // Edge - MLSD
  {
    name: "M-LSDPreprocessor",
    category: "Edge",
    types: { sd15: "mlsd", sdxl: "lineart" },
  },
  // Edge - Soft Edge
  {
    name: "HEDPreprocessor",
    category: "Edge",
    types: { sd15: "softedge", sdxl: "scribble" },
  },
  {
    name: "PiDiNetPreprocessor",
    category: "Edge",
    types: { sd15: "softedge", sdxl: "scribble" },
  },
  {
    name: "TEEDPreprocessor",
    category: "Edge",
    types: { sd15: "softedge", sdxl: "lineart" },
  },
  // Scribble
  {
    name: "ScribblePreprocessor",
    category: "Scribble",
    types: { sd15: "scribble", sdxl: "scribble" },
  },
  {
    name: "FakeScribblePreprocessor",
    category: "Scribble",
    types: { sd15: "scribble", sdxl: "scribble" },
  },
  {
    name: "Scribble_XDoG_Preprocessor",
    category: "Scribble",
    types: { sd15: "scribble", sdxl: "scribble" },
  },
  {
    name: "Scribble_PiDiNet_Preprocessor",
    category: "Scribble",
    types: { sd15: "scribble", sdxl: "scribble" },
  },
  // Segment
  {
    name: "SAMPreprocessor",
    category: "Segment",
    types: { sd15: "segment", sdxl: "segment" },
  },
  {
    name: "OneFormer-COCO-SemSegPreprocessor",
    category: "Segment",
    types: { sd15: "segment", sdxl: "segment" },
  },
  {
    name: "OneFormer-ADE20K-SemSegPreprocessor",
    category: "Segment",
    types: { sd15: "segment", sdxl: "segment" },
  },
  {
    name: "UniFormer-SemSegPreprocessor",
    category: "Segment",
    types: { sd15: "segment", sdxl: "segment" },
  },
  {
    name: "SemSegPreprocessor",
    category: "Segment",
    types: { sd15: "segment", sdxl: "segment" },
  },
  {
    name: "AnimeFace_SemSegPreprocessor",
    category: "Segment",
    types: { sd15: "segment", sdxl: "segment" },
  },
  // Tile
  {
    name: "TilePreprocessor",
    category: "Tile",
    types: { sd15: "tile", sdxl: "tile" },
  },
  {
    name: "TTPlanet_TileGF_Preprocessor",
    category: "Tile",
    types: { sd15: "tile", sdxl: "tile" },
  },
  {
    name: "TTPlanet_TileSimple_Preprocessor",
    category: "Tile",
    types: { sd15: "tile", sdxl: "tile" },
  },
  // Shuffle (SD1.5 only)
  {
    name: "ShufflePreprocessor",
    category: "Other",
    types: { sd15: "shuffle" },
  },
  // Repaint (SDXL only)
  {
    name: "ImageIntensityDetector",
    category: "Other",
    types: { sdxl: "repaint" },
  },
  {
    name: "ImageLuminanceDetector",
    category: "Other",
    types: { sdxl: "repaint" },
  },
  { name: "ColorPreprocessor", category: "Other", types: { sdxl: "repaint" } },
  // Inpaint (SD1.5 only)
  {
    name: "InpaintPreprocessor",
    category: "Other",
    types: { sd15: "inpaint" },
  },
];
