export class GraphicsCapabilities {
  private static instance: GraphicsCapabilities;
  private hasWebGPU: boolean = false;
  private hasWebGL2: boolean = false;
  
  private constructor() {}

  static async getInstance(): Promise<GraphicsCapabilities> {
    if (!GraphicsCapabilities.instance) {
      GraphicsCapabilities.instance = new GraphicsCapabilities();
      await GraphicsCapabilities.instance.initialize();
    }
    return GraphicsCapabilities.instance;
  }

  private async initialize(): Promise<void> {
    // Check WebGPU support
    if ('gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          const device = await adapter.requestDevice();
          this.hasWebGPU = !!device;
        }
      } catch (e) {
        console.warn('WebGPU not available:', e);
      }
    }

    // Check WebGL2 support
    try {
      const canvas = document.createElement('canvas');
      this.hasWebGL2 = !!canvas.getContext('webgl2');
    } catch (e) {
      console.warn('WebGL2 not available:', e);
    }
  }

  public getPreferredAPI(): 'webgpu' | 'webgl2' | 'none' {
    if (this.hasWebGPU) return 'webgpu';
    if (this.hasWebGL2) return 'webgl2';
    return 'none';
  }

  public getCapabilities() {
    return {
      webgpu: this.hasWebGPU,
      webgl2: this.hasWebGL2,
      preferredAPI: this.getPreferredAPI(),
    };
  }
}