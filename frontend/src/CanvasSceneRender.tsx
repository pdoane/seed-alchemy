import { CanvasElementState, CanvasStrokeState } from "./schema";
import { stateCanvas, stateSession, stateSystem } from "./store";
import { Vec2 } from "./vec2";

function hexToRgbFloat(hex: string) {
  hex = hex.substring(1);
  const r = parseInt(hex.slice(0, 2), 16) / 255;
  const g = parseInt(hex.slice(2, 4), 16) / 255;
  const b = parseInt(hex.slice(4, 6), 16) / 255;
  return [r, g, b];
}

function checkError(gl: WebGL2RenderingContext) {
  const error = gl.getError();
  if (error !== gl.NO_ERROR) {
    const errorMessage = `WebGL Error: ${error}`;
    console.error(errorMessage);
  }
}

function makeBuffer(gl: WebGL2RenderingContext): WebGLBuffer {
  const buffer = gl.createBuffer();
  if (!buffer) {
    throw new Error("createBuffer Failed");
  }
  return buffer;
}

function makeFramebuffer(gl: WebGL2RenderingContext): WebGLFramebuffer {
  const framebuffer = gl.createFramebuffer();
  if (!framebuffer) {
    throw new Error("createFramebuffer Failed");
  }

  return framebuffer;
}

function makeTexture(gl: WebGL2RenderingContext): WebGLTexture {
  const texture = gl.createTexture();
  if (!texture) {
    throw new Error("createTexture Failed");
  }
  return texture;
}

function makeShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error("createShader Failed");
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader)!);
  }
  return shader;
}

function makeProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string): WebGLProgram {
  let vertexShader = null;
  let fragmentShader = null;
  try {
    vertexShader = makeShader(gl, gl.VERTEX_SHADER, vsSource);
    fragmentShader = makeShader(gl, gl.FRAGMENT_SHADER, fsSource);

    const shaderProgram = gl.createProgram();
    if (!shaderProgram) {
      throw new Error("createProgram Failed");
    }
    gl.attachShader(shaderProgram, vertexShader);
    gl.attachShader(shaderProgram, fragmentShader);
    gl.linkProgram(shaderProgram);
    return shaderProgram;
  } finally {
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
  }
}

function makeVertexArrayObject(gl: WebGL2RenderingContext): WebGLVertexArrayObject {
  const vao = gl.createVertexArray();
  if (!vao) {
    throw new Error("createVertexArray Failed");
  }
  return vao;
}

function makeCheckerboardTexture(gl: WebGL2RenderingContext): WebGLTexture {
  const SIZE = 16;
  const PIXEL_SIZE = 8;
  const textureData = new Uint8Array(SIZE * SIZE * 4);

  for (let i = 0; i < SIZE; i++) {
    for (let j = 0; j < SIZE; j++) {
      let pixelIndex = (i * SIZE + j) * 4;
      let isWhiteSquare = (((i / PIXEL_SIZE) ^ (j / PIXEL_SIZE)) & 1) === 1;

      textureData[pixelIndex] = isWhiteSquare ? 211 : 169;
      textureData[pixelIndex + 1] = isWhiteSquare ? 211 : 169;
      textureData[pixelIndex + 2] = isWhiteSquare ? 211 : 169;
      textureData[pixelIndex + 3] = 255;
    }
  }

  const texture = makeTexture(gl);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, SIZE, SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, textureData);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.generateMipmap(gl.TEXTURE_2D);
  return texture;
}

function depthValue(index: number) {
  const MaxDepth = 65535; // 16-bit depth buffer should be fine for our use
  return 1.0 - index / MaxDepth;
}

class View {
  public translate: Vec2 = Vec2.create(0, 0);
  public scale = 1.0;
}

interface Context {
  gl: WebGL2RenderingContext;
  view: View;
  viewportWidth: number;
  viewportHeight: number;
  currentProgram: WebGLProgram | null;
}

const vsGrid = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_tc;
out vec2 v_tc;
void main() {
  v_tc = a_tc;
  gl_Position = vec4(a_tc.x * 2.0 - 1.0, a_tc.y * -2.0 + 1.0, 1.0, 1.0);
}`;

const fsGrid = `#version 300 es
precision highp float;
in vec2 v_tc;
out vec4 o_color;
uniform vec4 u_params;

// https://iquilezles.org/articles/filterableprocedurals/
float filteredGrid(in vec2 p, in vec2 dpdx, in vec2 dpdy)
{
    const float N = 64.0;
    vec2 w = max(abs(dpdx), abs(dpdy));
    vec2 a = p + 0.5*w;                        
    vec2 b = p - 0.5*w;           
    vec2 i = (floor(a)+min(fract(a)*N,1.0)-
              floor(b)-min(fract(b)*N,1.0))/(N*w);
    return (1.0-i.x)*(1.0-i.y);
}

void main() {
  vec2 uv = v_tc * u_params.xy + u_params.zw;
  vec2 ddx_uv = dFdx(uv);
  vec2 ddy_uv = dFdy(uv); 

  float grid = filteredGrid(uv, ddx_uv, ddy_uv);
  o_color.rgb = vec3(0.5) * (1.0 - grid);
  o_color.a = 1.0;
}`;

class GridRender {
  private shaderProgram: WebGLProgram;
  private vertexBuffer: WebGLBuffer;
  private vao: WebGLVertexArrayObject;
  private u_params: WebGLUniformLocation | null;

  constructor(gl: WebGL2RenderingContext) {
    this.shaderProgram = makeProgram(gl, vsGrid, fsGrid);
    this.u_params = gl.getUniformLocation(this.shaderProgram, "u_params");

    this.vao = makeVertexArrayObject(gl);
    gl.bindVertexArray(this.vao);

    this.vertexBuffer = makeBuffer(gl);
    const vertices = [0, 0, 2, 0, 0, 2];

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
  }

  cleanup(gl: WebGL2RenderingContext) {
    gl.deleteBuffer(this.vertexBuffer);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.shaderProgram);
  }

  setup(context: Context) {
    if (context.currentProgram == this.shaderProgram) return;
    const { gl, view, viewportWidth, viewportHeight } = context;
    const gridSize = 64;

    gl.useProgram(this.shaderProgram);
    gl.uniform4f(
      this.u_params,
      viewportWidth / view.scale / gridSize,
      viewportHeight / view.scale / gridSize,
      (0.5 - view.translate.x) / gridSize,
      (0.5 - view.translate.y) / gridSize
    );

    gl.bindVertexArray(this.vao);
    context.currentProgram = this.shaderProgram;
  }

  render(context: Context) {
    this.setup(context);
    const { gl } = context;

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

const vsImage = `#version 300 es
precision highp float;
uniform vec4 u_world;
uniform vec4 u_viewProj;
uniform float u_depth;
uniform vec4 u_tc;
layout(location=0) in vec2 a_tc;
out vec2 v_tc;
void main() {
  vec2 pos = a_tc * u_world.xy + u_world.zw;
  gl_Position = vec4(pos * u_viewProj.xy + u_viewProj.zw, u_depth, 1.0);
  v_tc = a_tc * u_tc.xy + u_tc.zw;
}`;

const fsImage = `#version 300 es
precision highp float;
uniform sampler2D u_texture;
in vec2 v_tc;
out vec4 o_color;

void main() {
  o_color = texture(u_texture, v_tc);
}`;

class ImageRender {
  private shaderProgram: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private indexBuffer: WebGLBuffer;
  private vertexBuffer: WebGLBuffer;
  private u_world: WebGLUniformLocation | null;
  private u_viewProj: WebGLUniformLocation | null;
  private u_depth: WebGLUniformLocation | null;
  private u_tc: WebGLUniformLocation | null;
  private u_texture: WebGLUniformLocation | null;

  constructor(gl: WebGL2RenderingContext) {
    this.shaderProgram = makeProgram(gl, vsImage, fsImage);
    this.u_world = gl.getUniformLocation(this.shaderProgram, "u_world");
    this.u_viewProj = gl.getUniformLocation(this.shaderProgram, "u_viewProj");
    this.u_depth = gl.getUniformLocation(this.shaderProgram, "u_depth");
    this.u_tc = gl.getUniformLocation(this.shaderProgram, "u_tc");
    this.u_texture = gl.getUniformLocation(this.shaderProgram, "u_texture");

    this.vao = makeVertexArrayObject(gl);
    gl.bindVertexArray(this.vao);

    this.indexBuffer = makeBuffer(gl);
    const indices = [0, 1, 2, 2, 1, 3];

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

    this.vertexBuffer = makeBuffer(gl);
    const vertices = [0, 0, 1, 0, 0, 1, 1, 1];

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
  }

  cleanup(gl: WebGL2RenderingContext) {
    gl.deleteBuffer(this.indexBuffer);
    gl.deleteBuffer(this.vertexBuffer);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.shaderProgram);
  }

  setup(context: Context) {
    if (context.currentProgram == this.shaderProgram) return;
    const { gl, view, viewportWidth, viewportHeight } = context;

    gl.useProgram(this.shaderProgram);
    gl.uniform4f(
      this.u_viewProj,
      (2 / viewportWidth) * view.scale,
      (-2 / viewportHeight) * view.scale,
      (2 / viewportWidth) * (view.translate.x * view.scale) - 1,
      (-2 / viewportHeight) * (view.translate.y * view.scale) + 1
    );
    gl.uniform1i(this.u_texture, 0);
    gl.bindVertexArray(this.vao);
    context.currentProgram = this.shaderProgram;
  }

  render(
    context: Context,
    index: number,
    element: CanvasElementState,
    texture: WebGLTexture | null,
    tc_sx: number,
    tc_sy: number,
    tc_dx: number,
    tc_dy: number
  ) {
    this.setup(context);
    const { gl } = context;

    gl.uniform4f(this.u_world, element.width, element.height, element.x, element.y);
    gl.uniform1f(this.u_depth, depthValue(index));
    gl.uniform4f(this.u_tc, tc_sx, tc_sy, tc_dx, tc_dy);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);

    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }
}

const vsSelection = `#version 300 es
precision highp float;
uniform vec4 u_world;
uniform vec4 u_viewProj;
uniform float u_depth;
uniform vec4 u_tc;
layout(location=0) in vec2 a_tc;
out vec2 v_tc;
void main() {
  vec2 pos = a_tc * u_world.xy + u_world.zw;
  gl_Position = vec4(pos * u_viewProj.xy + u_viewProj.zw, u_depth, 1.0);
  v_tc = a_tc * u_tc.xy + u_tc.zw;
}`;

const fsSelection = `#version 300 es
precision highp float;
uniform vec4 u_params;
uniform vec4 u_color;
in vec2 v_tc;
out vec4 o_color;

void main() {
  // pixel region
  vec2 p0 = abs(v_tc);
  vec2 p1 = p0 + vec2(u_params.z);

  // edge region
  vec2 e0 = u_params.xy - vec2(u_params.w);
  vec2 e1 = u_params.xy + vec2(u_params.w);

  // coverage
  vec2 c = max(min(p1, e1) - max(p0, e0), vec2(0.0)) / vec2(u_params.z);

  // side
  vec2 s = p0 - e0;

  // result
  float x = mix(c.x, c.y, s.x < s.y);
  o_color = u_color * x;
}`;

class SelectionRender {
  private shaderProgram: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private indexBuffer: WebGLBuffer;
  private vertexBuffer: WebGLBuffer;
  private u_world: WebGLUniformLocation | null;
  private u_viewProj: WebGLUniformLocation | null;
  private u_depth: WebGLUniformLocation | null;
  private u_tc: WebGLUniformLocation | null;
  private u_params: WebGLUniformLocation | null;
  private u_color: WebGLUniformLocation | null;

  constructor(gl: WebGL2RenderingContext) {
    this.shaderProgram = makeProgram(gl, vsSelection, fsSelection);
    this.u_world = gl.getUniformLocation(this.shaderProgram, "u_world");
    this.u_viewProj = gl.getUniformLocation(this.shaderProgram, "u_viewProj");
    this.u_depth = gl.getUniformLocation(this.shaderProgram, "u_depth");
    this.u_tc = gl.getUniformLocation(this.shaderProgram, "u_tc");
    this.u_params = gl.getUniformLocation(this.shaderProgram, "u_params");
    this.u_color = gl.getUniformLocation(this.shaderProgram, "u_color");

    this.vao = makeVertexArrayObject(gl);
    gl.bindVertexArray(this.vao);

    this.indexBuffer = makeBuffer(gl);
    const indices = [0, 1, 2, 2, 1, 3];

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

    this.vertexBuffer = makeBuffer(gl);
    const vertices = [0, 0, 1, 0, 0, 1, 1, 1];

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
  }

  cleanup(gl: WebGL2RenderingContext) {
    gl.deleteBuffer(this.indexBuffer);
    gl.deleteBuffer(this.vertexBuffer);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.shaderProgram);
  }

  setup(context: Context) {
    if (context.currentProgram == this.shaderProgram) return;
    const { gl, view, viewportWidth, viewportHeight } = context;

    gl.useProgram(this.shaderProgram);
    gl.uniform4f(
      this.u_viewProj,
      (2 / viewportWidth) * view.scale,
      (-2 / viewportHeight) * view.scale,
      (2 / viewportWidth) * (view.translate.x * view.scale) - 1,
      (-2 / viewportHeight) * (view.translate.y * view.scale) + 1
    );

    gl.bindVertexArray(this.vao);
    context.currentProgram = this.shaderProgram;
  }

  render(context: Context, index: number, element: CanvasElementState, color: string) {
    this.setup(context);
    const { gl, view } = context;

    const t = 2;
    const [r, g, b] = hexToRgbFloat(color);
    gl.uniform4f(this.u_world, element.width + 2 * t, element.height + 2 * t, element.x - t, element.y - t);
    gl.uniform1f(this.u_depth, depthValue(index));
    gl.uniform4f(
      this.u_tc,
      element.width + 2 * t,
      element.height + 2 * t,
      -element.width / 2 - t,
      -element.height / 2 - t
    );
    gl.uniform4f(this.u_params, element.width / 2, element.height / 2, 1 / view.scale, t);
    gl.uniform4f(this.u_color, r, g, b, 1);

    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }
}

const vsBrush = `#version 300 es
precision highp float;
uniform vec4 u_world;
uniform vec4 u_viewProj;
uniform vec4 u_tc;
layout(location=0) in vec2 a_tc;
out vec2 v_tc;
void main() {
  vec2 pos = a_tc * u_world.xy + u_world.zw;
  gl_Position = vec4(pos * u_viewProj.xy + u_viewProj.zw, 0.0, 1.0);
  v_tc = a_tc * u_tc.xy + u_tc.zw;
}`;

const fsBrush = `#version 300 es
precision highp float;
uniform vec4 u_params;
in vec2 v_tc;
out vec4 o_color;

void main() {
  float e0 = u_params.x + u_params.w;
  float e1 = u_params.x - u_params.w;

  // 16x SSAA
  vec2 offsets[] = vec2[](
    vec2( 1.0/8.0,  1.0/8.0),
    vec2(-1.0/8.0, -3.0/8.0),
    vec2(-3.0/8.0,  2.0/8.0),
    vec2( 4.0/8.0, -1.0/8.0),
    vec2(-5.0/8.0, -2.0/8.0),
    vec2( 2.0/8.0,  5.0/8.0),
    vec2( 5.0/8.0,  3.0/8.0),
    vec2( 3.0/8.0, -5.0/8.0),
    vec2(-2.0/8.0,  6.0/8.0),
    vec2( 0.0/8.0, -7.0/8.0),
    vec2(-4.0/8.0, -6.0/8.0),
    vec2(-6.0/8.0,  4.0/8.0),
    vec2(-8.0/8.0,  0.0/8.0),
    vec2( 7.0/8.0, -4.0/8.0),
    vec2( 6.0/8.0,  7.0/8.0),
    vec2(-7.0/8.0, -8.0/8.0)
  );

  float x = 0.0;
  for (int i = 0; i < 16; ++i) {
    float r = length(v_tc + offsets[i] * u_params.z);
    float s = step(r, e0) - step(r, e1);
    x += s;
  }

  x *= 1.0 / 16.0;

  o_color = vec4(x) * 0.5;
}`;

class BrushRender {
  private shaderProgram: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private indexBuffer: WebGLBuffer;
  private vertexBuffer: WebGLBuffer;
  private u_world: WebGLUniformLocation | null;
  private u_viewProj: WebGLUniformLocation | null;
  private u_tc: WebGLUniformLocation | null;
  private u_params: WebGLUniformLocation | null;

  constructor(gl: WebGL2RenderingContext) {
    this.shaderProgram = makeProgram(gl, vsBrush, fsBrush);
    this.u_world = gl.getUniformLocation(this.shaderProgram, "u_world");
    this.u_viewProj = gl.getUniformLocation(this.shaderProgram, "u_viewProj");
    this.u_tc = gl.getUniformLocation(this.shaderProgram, "u_tc");
    this.u_params = gl.getUniformLocation(this.shaderProgram, "u_params");

    this.vao = makeVertexArrayObject(gl);
    gl.bindVertexArray(this.vao);

    this.indexBuffer = makeBuffer(gl);
    const indices = [0, 1, 2, 2, 1, 3];

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

    this.vertexBuffer = makeBuffer(gl);
    const vertices = [0, 0, 1, 0, 0, 1, 1, 1];

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
  }

  cleanup(gl: WebGL2RenderingContext) {
    gl.deleteBuffer(this.indexBuffer);
    gl.deleteBuffer(this.vertexBuffer);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.shaderProgram);
  }

  setup(context: Context) {
    if (context.currentProgram == this.shaderProgram) return;
    const { gl, view, viewportWidth, viewportHeight } = context;

    gl.useProgram(this.shaderProgram);
    gl.uniform4f(
      this.u_viewProj,
      (2 / viewportWidth) * view.scale,
      (-2 / viewportHeight) * view.scale,
      (2 / viewportWidth) * (view.translate.x * view.scale) - 1,
      (-2 / viewportHeight) * (view.translate.y * view.scale) + 1
    );

    gl.bindVertexArray(this.vao);
    context.currentProgram = this.shaderProgram;
  }

  render(context: Context, cursor: Vec2) {
    this.setup(context);
    const { gl, view } = context;

    const brushWidth = 64;
    const brushHeight = 64;
    const t = 2;
    gl.uniform4f(
      this.u_world,
      brushWidth + 2 * t,
      brushHeight + 2 * t,
      cursor.x - brushWidth / 2 - t,
      cursor.y - brushHeight / 2 - t
    );
    gl.uniform4f(this.u_tc, brushWidth + 2 * t, brushHeight + 2 * t, -brushWidth / 2 - t, -brushHeight / 2 - t);
    gl.uniform4f(this.u_params, brushWidth / 2, brushHeight / 2, 1 / view.scale, t);
    gl.activeTexture(gl.TEXTURE0);

    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }
}

// Instanced Line Rendering Part I - Rye Terrell
// https://wwwtyro.net/2019/11/18/instanced-lines.html
const vsStroke = `#version 300 es
precision highp float;
uniform vec4 u_viewProj;
uniform float u_thickness;
layout(location=0) in vec3 a_pos;
layout(location=1) in vec2 a_point0;
layout(location=2) in vec2 a_point1;

void main() {
  vec2 xBasis = normalize(a_point1 - a_point0);
  vec2 yBasis = vec2(-xBasis.y, xBasis.x);
  vec2 offset0 = a_point0 + u_thickness * (a_pos.x * xBasis + a_pos.y * yBasis);
  vec2 offset1 = a_point1 + u_thickness * (a_pos.x * xBasis + a_pos.y * yBasis);
  vec2 point = mix(offset0, offset1, a_pos.z);
  gl_Position = vec4(point * u_viewProj.xy + u_viewProj.zw, 0.0, 1.0);
}`;

const fsStroke = `#version 300 es
precision highp float;
uniform vec4 u_color;
out vec4 o_color;

void main() {
  o_color = u_color;
}`;

class StrokeRender {
  private shaderProgram: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private vertexBuffer: WebGLBuffer;
  private vertexCount: number;
  private positionBuffer: WebGLBuffer;
  private u_viewProj: WebGLUniformLocation | null;
  private u_thickness: WebGLUniformLocation | null;
  private u_color: WebGLUniformLocation | null;

  constructor(gl: WebGL2RenderingContext) {
    this.shaderProgram = makeProgram(gl, vsStroke, fsStroke);
    this.u_viewProj = gl.getUniformLocation(this.shaderProgram, "u_viewProj");
    this.u_thickness = gl.getUniformLocation(this.shaderProgram, "u_thickness");
    this.u_color = gl.getUniformLocation(this.shaderProgram, "u_color");

    this.vertexBuffer = makeBuffer(gl);
    this.positionBuffer = makeBuffer(gl);

    this.vao = makeVertexArrayObject(gl);
    gl.bindVertexArray(this.vao);

    this.vertexBuffer = makeBuffer(gl);

    const resolution = 16;
    const vertices = [0, -0.5, 0, 0, -0.5, 1, 0, 0.5, 1, 0, -0.5, 0, 0, 0.5, 1, 0, 0.5, 0];
    for (let step = 0; step < resolution; step++) {
      const theta0 = Math.PI / 2 + ((step + 0) * Math.PI) / resolution;
      const theta1 = Math.PI / 2 + ((step + 1) * Math.PI) / resolution;
      vertices.push(0, 0, 0);
      vertices.push(0.5 * Math.cos(theta0), 0.5 * Math.sin(theta0), 0);
      vertices.push(0.5 * Math.cos(theta1), 0.5 * Math.sin(theta1), 0);
    }
    for (let step = 0; step < resolution; step++) {
      const theta0 = (3 * Math.PI) / 2 + ((step + 0) * Math.PI) / resolution;
      const theta1 = (3 * Math.PI) / 2 + ((step + 1) * Math.PI) / resolution;
      vertices.push(0, 0, 1);
      vertices.push(0.5 * Math.cos(theta0), 0.5 * Math.sin(theta0), 1);
      vertices.push(0.5 * Math.cos(theta1), 0.5 * Math.sin(theta1), 1);
    }
    this.vertexCount = vertices.length / 3;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribDivisor(0, 0);
  }

  cleanup(gl: WebGL2RenderingContext) {
    gl.deleteBuffer(this.vertexBuffer);
    gl.deleteBuffer(this.positionBuffer);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.shaderProgram);
  }

  setup(context: Context) {
    if (context.currentProgram == this.shaderProgram) return;
    const { gl, view, viewportWidth, viewportHeight } = context;

    gl.useProgram(this.shaderProgram);
    gl.uniform4f(
      this.u_viewProj,
      (2 / viewportWidth) * view.scale,
      (-2 / viewportHeight) * view.scale,
      (2 / viewportWidth) * (view.translate.x * view.scale) - 1,
      (-2 / viewportHeight) * (view.translate.y * view.scale) + 1
    );

    gl.bindVertexArray(this.vao);
    context.currentProgram = this.shaderProgram;
  }

  render(context: Context, stroke: CanvasStrokeState) {
    const instanceCount = stroke.segments.length / 2 - 1;
    if (instanceCount < 1) return;

    this.setup(context);
    const { gl } = context;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(stroke.segments), gl.DYNAMIC_DRAW);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 8);
    gl.enableVertexAttribArray(1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribDivisor(1, 1);
    gl.vertexAttribDivisor(2, 1);

    gl.uniform1f(this.u_thickness, 64);
    if (stroke.tool === "brush") {
      gl.uniform4f(this.u_color, 0.0, 1.0, 1.0, 0.0);
    } else {
      gl.uniform4f(this.u_color, 1.0, 1.0, 1.0, 1.0);
    }

    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, this.vertexCount, instanceCount);
  }
}

const vsComposite = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_tc;
out vec2 v_tc;
void main() {
  v_tc = a_tc;
  gl_Position = vec4(a_tc.x * 2.0 - 1.0, a_tc.y * 2.0 - 1.0, 0.0, 1.0);
}`;

const fsComposite = `#version 300 es
precision highp float;
uniform sampler2D u_texture;
in vec2 v_tc;
out vec4 o_color;

void main() {
  vec4 t = vec4(1) - texture(u_texture, v_tc);
  float a = t.w * 0.5;
  o_color = vec4(t.xyz * a, a);
}`;

class CompositeRender {
  private shaderProgram: WebGLProgram;
  private vertexBuffer: WebGLBuffer;
  private vao: WebGLVertexArrayObject;
  private u_texture: WebGLUniformLocation | null;

  constructor(gl: WebGL2RenderingContext) {
    this.shaderProgram = makeProgram(gl, vsComposite, fsComposite);
    this.u_texture = gl.getUniformLocation(this.shaderProgram, "u_texture");

    this.vao = makeVertexArrayObject(gl);
    gl.bindVertexArray(this.vao);

    this.vertexBuffer = makeBuffer(gl);
    const vertices = [0, 0, 2, 0, 0, 2];

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
  }

  cleanup(gl: WebGL2RenderingContext) {
    gl.deleteBuffer(this.vertexBuffer);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.shaderProgram);
  }

  setup(context: Context) {
    if (context.currentProgram == this.shaderProgram) return;
    const { gl } = context;

    gl.useProgram(this.shaderProgram);
    gl.uniform1i(this.u_texture, 0);
    gl.bindVertexArray(this.vao);
    context.currentProgram = this.shaderProgram;
  }

  render(context: Context, maskTexture: WebGLTexture) {
    this.setup(context);
    const { gl } = context;

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, maskTexture);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindTexture(gl.TEXTURE_2D, null);
  }
}

interface SceneTexture {
  url: string | null;
  image: HTMLImageElement;
  texture: WebGLTexture | null;
}

export class CanvasSceneRender {
  public gl: WebGL2RenderingContext;
  private requestRender: () => void;
  private gridRender: GridRender;
  private imageRender: ImageRender;
  private selectionRender: SelectionRender;
  private brushRender: BrushRender;
  private strokeRender: StrokeRender;
  private compositeRender: CompositeRender;
  private lastWidth: number;
  private lastHeight: number;
  private maskTexture: WebGLTexture | null;
  private maskFbo: WebGLFramebuffer | null;
  private textures: Map<string, SceneTexture>;
  private frameTextures: Set<string>;
  private checkerboardTexture: WebGLTexture;
  private previewUrl: string | null;

  constructor(canvas: HTMLCanvasElement, requestRender: () => void) {
    const gl = canvas.getContext("webgl2", { depth: true });
    if (!gl) {
      throw new Error("Unable to initialize WebGL. Your browser or machine may not support it.");
    }
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    this.gl = gl;
    this.requestRender = requestRender;
    this.gridRender = new GridRender(gl);
    this.imageRender = new ImageRender(gl);
    this.selectionRender = new SelectionRender(gl);
    this.brushRender = new BrushRender(gl);
    this.strokeRender = new StrokeRender(gl);
    this.compositeRender = new CompositeRender(gl);
    this.lastWidth = 0;
    this.lastHeight = 0;
    this.maskTexture = null;
    this.maskFbo = null;
    this.textures = new Map<string, SceneTexture>();
    this.frameTextures = new Set<string>();
    this.checkerboardTexture = makeCheckerboardTexture(gl);
    this.previewUrl = null;
  }

  cleanup() {
    const gl = this.gl;
    if (gl) {
      this.clearTextureCache();

      if (this.maskFbo) gl.deleteFramebuffer(this.maskFbo);
      if (this.maskTexture) gl.deleteTexture(this.maskTexture);
      gl.deleteTexture(this.checkerboardTexture);

      this.brushRender.cleanup(gl);
      this.selectionRender.cleanup(gl);
      this.imageRender.cleanup(gl);
      this.gridRender.cleanup(gl);
    }
  }

  clearTextureCache() {
    const gl = this.gl;
    for (const sceneTexture of this.textures.values()) {
      gl.deleteTexture(sceneTexture.texture);
    }
    this.textures.clear();
  }

  setPreviewUrl(url: string | null) {
    this.previewUrl = url;
    this.requestRender();
  }

  synchronize() {
    const gl = this.gl;

    this.frameTextures.clear();
    for (const element of stateCanvas.elements) {
      this.frameTextures.add(element.id);
      let sceneTexture = this.textures.get(element.id);
      if (sceneTexture === undefined) {
        sceneTexture = { url: null, image: new Image(), texture: null };
        this.textures.set(element.id, sceneTexture);
      }

      let url = null;
      if (this.previewUrl && stateSession.generatorId == element.id) {
        url = this.previewUrl;
      } else if (element.imageIndex < element.images.length) {
        const imagePath = element.images[element.imageIndex].path;
        url = `images/${stateSystem.user}/${imagePath}`;
      }

      if (sceneTexture.url != url) {
        sceneTexture.url = url;
        if (url) {
          sceneTexture.image.src = url;
          sceneTexture.image.onload = () => {
            if (!sceneTexture) return;

            if (sceneTexture.texture) {
              gl.deleteTexture(sceneTexture.texture);
            }
            sceneTexture.texture = makeTexture(gl);
            gl.bindTexture(gl.TEXTURE_2D, sceneTexture.texture);

            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sceneTexture.image);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            this.requestRender();
          };
        }
      }
    }

    for (const [key, sceneTexture] of this.textures.entries()) {
      if (!this.frameTextures.has(key)) {
        gl.deleteTexture(sceneTexture.texture);
        this.textures.delete(key);
      }
    }
  }

  render() {
    const gl = this.gl;
    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;

    this.synchronize();

    const view: View = { translate: stateCanvas.translate, scale: stateCanvas.scale };
    const context = {
      gl,
      view,
      viewportWidth: width,
      viewportHeight: height,
      currentProgram: null,
    };

    // delete invalid fbos
    if (this.lastWidth != width || this.lastHeight != height) {
      gl.deleteFramebuffer(this.maskFbo);
      gl.deleteTexture(this.maskTexture);
      this.maskFbo = null;
      this.maskTexture = null;
      this.lastWidth = width;
      this.lastHeight = height;
    }

    // mask render
    if (this.maskTexture === null) {
      this.maskTexture = makeTexture(gl);
      gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    }

    if (this.maskFbo === null) {
      this.maskFbo = makeFramebuffer(gl);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.maskFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.maskTexture, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        console.error("Framebuffer is not complete!");
      }
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.maskFbo);
    gl.viewport(0, 0, width, height);

    gl.disable(gl.BLEND);

    gl.clearColor(1.0, 1.0, 1.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    for (const stroke of stateCanvas.strokes) {
      this.strokeRender.render(context, stroke);
    }

    // main render
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);

    gl.clear(gl.DEPTH_BUFFER_BIT);

    gl.disable(gl.BLEND);
    gl.depthMask(true);
    this.gridRender.render(context);
    for (let i = 0; i < stateCanvas.elements.length; i++) {
      const element = stateCanvas.elements[stateCanvas.elements.length - i - 1];
      const sceneTexture = this.textures.get(element.id);
      if (sceneTexture && sceneTexture.url) {
        this.imageRender.render(context, i + 1, element, sceneTexture.texture, 1.0, 1.0, 0.0, 0.0);
      } else {
        this.imageRender.render(
          context,
          0,
          element,
          this.checkerboardTexture,
          element.width / 16.0,
          element.height / 16.0,
          (element.x % 16) / 16.0,
          (element.y % 16) / 16.0
        );
      }
    }

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    for (let i = 0; i < stateCanvas.elements.length; i++) {
      const element = stateCanvas.elements[stateCanvas.elements.length - i - 1];
      if (stateCanvas.selectedId == element.id || stateCanvas.hoveredId == element.id) {
        const color = stateCanvas.hoveredId == element.id ? "#3B82F6" : "#2563EB";
        this.selectionRender.render(context, i, element, color);
      }
    }

    this.compositeRender.render(context, this.maskTexture);

    if (stateCanvas.cursorPos) {
      this.brushRender.render(context, stateCanvas.cursorPos);
    }

    checkError(gl);
  }

  generateMask(element: CanvasElementState): HTMLCanvasElement {
    const gl = this.gl;
    const width = element.width;
    const height = element.height;

    // Framebuffer
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      console.error("Framebuffer is not complete!");
    }

    // Render
    gl.disable(gl.BLEND);
    gl.viewport(0, 0, width, height);

    const view: View = { translate: Vec2.create(-element.x, -element.y), scale: 1.0 };
    const context = {
      gl,
      view,
      viewportWidth: width,
      viewportHeight: height,
      currentProgram: null,
    };

    for (let i = 0; i < stateCanvas.elements.length; i++) {
      const element = stateCanvas.elements[stateCanvas.elements.length - i - 1];
      const sceneTexture = this.textures.get(element.id);
      if (sceneTexture && sceneTexture.url) {
        this.imageRender.render(context, i, element, sceneTexture.texture, 1.0, 1.0, 0.0, 0.0);
      }
    }
    gl.colorMask(false, false, false, true);
    for (const stroke of stateCanvas.strokes) {
      this.strokeRender.render(context, stroke);
    }
    gl.colorMask(true, true, true, true);

    const flippedData = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, flippedData);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    checkError(gl);

    // Flip the pixels on the Y-axis
    const imageData = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcIndex = (y * width + x) * 4;
        const destIndex = ((height - y - 1) * width + x) * 4;
        imageData.set(flippedData.subarray(srcIndex, srcIndex + 4), destIndex);
      }
    }

    // Blob
    const offscreenCanvas = document.createElement("canvas");
    offscreenCanvas.width = width;
    offscreenCanvas.height = height;
    const ctx = offscreenCanvas.getContext("2d");
    if (!ctx) {
      throw new Error("getContext Failed");
    }
    const imgData = ctx.createImageData(width, height);
    imgData.data.set(imageData);
    ctx.putImageData(imgData, 0, 0);
    return offscreenCanvas;
  }
}
