import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { GraphicsCapabilities } from './GraphicsCapabilities';

export class VisualizationManager {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private animationObjects: { update: () => void }[] = [];

  constructor(container: HTMLElement) {
    // Initialize Three.js scene
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    
    // Setup renderer
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    // Setup camera and controls
    this.camera.position.z = 5;
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(10, 10, 10);
    this.scene.add(directionalLight);

    // Start animation loop
    this.animate();

    // Handle window resize
    window.addEventListener('resize', () => this.handleResize(container));
  }

  private handleResize(container: HTMLElement) {
    const width = container.clientWidth;
    const height = container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  private animate = () => {
    requestAnimationFrame(this.animate);
    
    // Update controls
    this.controls.update();

    // Update all animation objects
    this.animationObjects.forEach(obj => obj.update());

    // Render scene
    this.renderer.render(this.scene, this.camera);
  }

  // Add a visual representation of AI response
  public visualizeResponse(response: string) {
    // Clear existing visualization
    while(this.scene.children.length > 0) { 
      const obj = this.scene.children[0];
      if (obj instanceof THREE.Light) continue;
      this.scene.remove(obj);
    }

    // Create visualization based on response
    if (response.includes('error') || response.includes('failed')) {
      this.addErrorVisualization();
    } else {
      this.addSuccessVisualization();
    }
  }

  private addErrorVisualization() {
    const geometry = new THREE.IcosahedronGeometry(1, 0);
    const material = new THREE.MeshPhongMaterial({ 
      color: 0xff0000,
      wireframe: true,
      transparent: true,
      opacity: 0.8
    });
    const mesh = new THREE.Mesh(geometry, material);
    
    this.scene.add(mesh);
    this.animationObjects.push({
      update: () => {
        mesh.rotation.x += 0.01;
        mesh.rotation.y += 0.01;
      }
    });
  }

  private addSuccessVisualization() {
    const geometry = new THREE.TorusKnotGeometry(1, 0.3, 100, 16);
    const material = new THREE.MeshPhongMaterial({ 
      color: 0x00ff00,
      transparent: true,
      opacity: 0.8
    });
    const mesh = new THREE.Mesh(geometry, material);
    
    this.scene.add(mesh);
    this.animationObjects.push({
      update: () => {
        mesh.rotation.x += 0.01;
        mesh.rotation.y += 0.01;
      }
    });
  }

  // Add custom visualization
  public addCustomVisualization(type: string, data: any) {
    switch (type) {
      case 'barChart':
        this.createBarChart(data);
        break;
      case 'sphereField':
        this.createSphereField(data);
        break;
      case 'networkGraph':
        this.createNetworkGraph(data);
        break;
      default:
        console.warn('Unknown visualization type:', type);
    }
  }

  private createBarChart(data: number[]) {
    const group = new THREE.Group();
    
    data.forEach((value, index) => {
      const height = value * 2;
      const geometry = new THREE.BoxGeometry(0.5, height, 0.5);
      const material = new THREE.MeshPhongMaterial({ 
        color: new THREE.Color().setHSL(index / data.length, 1, 0.5) 
      });
      const bar = new THREE.Mesh(geometry, material);
      
      bar.position.x = index - (data.length - 1) / 2;
      bar.position.y = height / 2;
      
      group.add(bar);
    });

    this.scene.add(group);
  }

  private createSphereField(count: number) {
    const group = new THREE.Group();
    
    for (let i = 0; i < count; i++) {
      const geometry = new THREE.SphereGeometry(0.2, 32, 32);
      const material = new THREE.MeshPhongMaterial({ 
        color: new THREE.Color().setHSL(Math.random(), 1, 0.5) 
      });
      const sphere = new THREE.Mesh(geometry, material);
      
      sphere.position.x = (Math.random() - 0.5) * 10;
      sphere.position.y = (Math.random() - 0.5) * 10;
      sphere.position.z = (Math.random() - 0.5) * 10;
      
      group.add(sphere);
    }

    this.scene.add(group);
  }

  private createNetworkGraph(connections: [number, number][]) {
    const group = new THREE.Group();
    const nodes: THREE.Mesh[] = [];
    
    // Create nodes
    for (let i = 0; i < 10; i++) {
      const geometry = new THREE.SphereGeometry(0.2, 32, 32);
      const material = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
      const node = new THREE.Mesh(geometry, material);
      
      node.position.x = (Math.random() - 0.5) * 10;
      node.position.y = (Math.random() - 0.5) * 10;
      node.position.z = (Math.random() - 0.5) * 10;
      
      nodes.push(node);
      group.add(node);
    }
    
    // Create connections
    connections.forEach(([from, to]) => {
      const points = [nodes[from].position, nodes[to].position];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: 0xffffff });
      const line = new THREE.Line(geometry, material);
      group.add(line);
    });

    this.scene.add(group);
  }
}