import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass';

const BackgroundVisualization: React.FC = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const composerRef = useRef<EffectComposer | null>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        // Initialize scene
        const scene = new THREE.Scene();
        sceneRef.current = scene;

        // Initialize camera
        const camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        camera.position.z = 5;
        cameraRef.current = camera;

        // Initialize renderer
        const renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
        });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Post-processing
        const composer = new EffectComposer(renderer);
        const renderPass = new RenderPass(scene, camera);
        composer.addPass(renderPass);

        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            1.5,
            0.4,
            0.85
        );
        composer.addPass(bloomPass);
        composerRef.current = composer;

        // Create particle system
        const particles = createParticleSystem();
        scene.add(particles);

        // Create nebula effect
        const nebula = createNebulaEffect();
        scene.add(nebula);

        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);
            if (particles) {
                particles.rotation.x += 0.0003;
                particles.rotation.y += 0.0005;
            }
            if (nebula) {
                nebula.rotation.z += 0.0001;
            }
            composer.render();
        };
        animate();

        // Handle resize
        const handleResize = () => {
            if (!renderer || !camera || !composer) return;
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            composer.setSize(window.innerWidth, window.innerHeight);
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            if (containerRef.current && renderer) {
                containerRef.current.removeChild(renderer.domElement);
            }
        };
    }, []);

    const createParticleSystem = () => {
        const geometry = new THREE.BufferGeometry();
        const particles = 5000;
        const positions = new Float32Array(particles * 3);
        const colors = new Float32Array(particles * 3);

        for (let i = 0; i < particles * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 10;
            positions[i + 1] = (Math.random() - 0.5) * 10;
            positions[i + 2] = (Math.random() - 0.5) * 10;

            colors[i] = Math.random();
            colors[i + 1] = Math.random();
            colors[i + 2] = Math.random();
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            transparent: true,
            blending: THREE.AdditiveBlending,
        });

        return new THREE.Points(geometry, material);
    };

    const createNebulaEffect = () => {
        const geometry = new THREE.PlaneGeometry(20, 20, 100, 100);
        const material = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec2 resolution;
                varying vec2 vUv;

                float noise(vec2 p) {
                    return fract(sin(dot(p.xy, vec2(12.9898,78.233))) * 43758.5453123);
                }

                void main() {
                    vec2 uv = vUv;
                    float n = noise(uv * 10.0);
                    vec3 color = vec3(0.5 + 0.5 * sin(time + uv.x),
                                    0.5 + 0.5 * cos(time + uv.y),
                                    0.5 + 0.5 * sin(time + n));
                    gl_FragColor = vec4(color, 0.5);
                }
            `,
            transparent: true,
            blending: THREE.AdditiveBlending,
        });

        return new THREE.Mesh(geometry, material);
    };

    return <div ref={containerRef} style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }} />;
};

export default BackgroundVisualization;