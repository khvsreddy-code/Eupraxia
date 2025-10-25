export class ShaderManager {
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private pipeline: GPURenderPipeline | null = null;

  private vertexShader = `
    struct VertexOutput {
      @builtin(position) position: vec4f,
      @location(0) color: vec4f,
    }

    @vertex
    fn main(@location(0) position: vec3f,
            @location(1) color: vec4f) -> VertexOutput {
      var output: VertexOutput;
      output.position = vec4f(position, 1.0);
      output.color = color;
      return output;
    }
  `;

  private fragmentShader = `
    @fragment
    fn main(@location(0) color: vec4f) -> @location(0) vec4f {
      return color;
    }
  `;

  async initialize(canvas: HTMLCanvasElement): Promise<boolean> {
    if (!navigator.gpu) {
      console.warn('WebGPU not supported');
      return false;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.warn('No appropriate GPUAdapter found');
        return false;
      }

      this.device = await adapter.requestDevice();
      this.context = canvas.getContext('webgpu');

      if (!this.context) {
        console.warn('Failed to get WebGPU context');
        return false;
      }

      const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({
        device: this.device,
        format: canvasFormat,
        alphaMode: 'premultiplied',
      });

      this.pipeline = this.device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: this.device.createShaderModule({
            code: this.vertexShader,
          }),
          entryPoint: 'main',
          buffers: [
            {
              arrayStride: 28,
              attributes: [
                {
                  shaderLocation: 0,
                  offset: 0,
                  format: 'float32x3',
                },
                {
                  shaderLocation: 1,
                  offset: 12,
                  format: 'float32x4',
                },
              ],
            },
          ],
        },
        fragment: {
          module: this.device.createShaderModule({
            code: this.fragmentShader,
          }),
          entryPoint: 'main',
          targets: [
            {
              format: canvasFormat,
            },
          ],
        },
        primitive: {
          topology: 'triangle-list',
        },
      });

      return true;
    } catch (error) {
      console.error('Failed to initialize WebGPU:', error);
      return false;
    }
  }

  createBuffer(data: Float32Array, usage: GPUBufferUsageFlags): GPUBuffer | null {
    if (!this.device) return null;

    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage,
      mappedAtCreation: true,
    });

    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();

    return buffer;
  }

  async render(vertexBuffer: GPUBuffer, colorBuffer: GPUBuffer, vertexCount: number): Promise<void> {
    if (!this.device || !this.context || !this.pipeline) return;

    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setVertexBuffer(0, vertexBuffer);
    passEncoder.setVertexBuffer(1, colorBuffer);
    passEncoder.draw(vertexCount, 1, 0, 0);
    passEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  // Create a compute pipeline for data processing
  async createComputePipeline(shaderCode: string, entryPoint: string): Promise<GPUComputePipeline | null> {
    if (!this.device) return null;

    try {
      return this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: this.device.createShaderModule({
            code: shaderCode,
          }),
          entryPoint: entryPoint,
        },
      });
    } catch (error) {
      console.error('Failed to create compute pipeline:', error);
      return null;
    }
  }

  // Run a compute shader
  async dispatch(
    pipeline: GPUComputePipeline,
    inputBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    workgroupCount: [number, number, number]
  ): Promise<void> {
    if (!this.device) return;

    const bindGroupLayout = pipeline.getBindGroupLayout(0);
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: { buffer: inputBuffer },
        },
        {
          binding: 1,
          resource: { buffer: outputBuffer },
        },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(...workgroupCount);
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }
}