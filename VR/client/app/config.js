import * as THREE from "three";

export function createOctoBasePositions() {
  const centerX = 0.0;
  const centerY = 1.0;
  const centerZ = -0.15;
  const radius = 0.32;
  const angleOffset = -0.5 * Math.PI;
  return Array.from({ length: 8 }, (_, index) =>
    new THREE.Vector3(
      centerX + radius * Math.cos(angleOffset + (2.0 * Math.PI * index) / 8.0),
      centerY,
      centerZ + radius * Math.sin(angleOffset + (2.0 * Math.PI * index) / 8.0)
    )
  );
}

export function isEightArmOctoMode(modeKey) {
  return modeKey === "cathy-foraging" || modeKey === "octo-waypoint";
}

export function createConfig({ window }) {
  const searchParams = new URLSearchParams(window.location.search);
  const isDev = searchParams.has("dev");
  const preferInsecureWebSocket =
    searchParams.get("ws_insecure") === "1" ||
    searchParams.get("ws_insecure") === "true";
  const defaultWsScheme =
    preferInsecureWebSocket || window.location.protocol !== "https:" ? "ws" : "wss";
  const defaultWsHost = window.location.hostname || "127.0.0.1";
  const defaultWsPort = searchParams.get("ws_port") ?? "8765";
  const serverHost =
    searchParams.get("ws") ??
    `${defaultWsScheme}://${defaultWsHost}:${defaultWsPort}`;
  const fallbackServerHost =
    serverHost.startsWith("wss://") && preferInsecureWebSocket
      ? serverHost.replace(/^wss:\/\//, "ws://")
      : null;

  const calibrationOffset = new THREE.Vector3(
    Number(searchParams.get("ox") ?? 0.0),
    Number(searchParams.get("oy") ?? 0.0),
    Number(searchParams.get("oz") ?? 0.0)
  );

  const cathyForagingBasePositions = createOctoBasePositions();
  const demoArmIds = [
    "left_arm",
    "right_arm"
  ];

  const armConfig = {
    left_arm: {
      color: "#ff6b6b",
      base: new THREE.Vector3(-0.25, 0.95, -0.35),
      hand: "left",
    },
    right_arm: {
      color: "#74c0fc",
      base: new THREE.Vector3(0.25, 0.95, -0.35),
      hand: "right",
    }
  };

  const characterModes = {
    "demo-spline": {
      allowsBaseControl: true,
      leftBase: new THREE.Vector3(-0.25, 0.95, -0.35),
      rightBase: new THREE.Vector3(0.25, 0.95, -0.35),
    },
    "two-cr": {
      allowsBaseControl: false,
      leftBase: new THREE.Vector3(-0.15, 1.0, -0.15),
      rightBase: new THREE.Vector3(0.15, 1.0, -0.15),
    }
  };

  return {
    searchParams,
    isDev,
    serverHost,
    fallbackServerHost,
    preferInsecureWebSocket,
    calibrationOffset,
    initialControllerForward,
    demoArmIds,
    armIds: demoArmIds,
    armConfig,
    characterModes,
    curveSamples: 21,
    tipDefaultRadius: 0.045,
    maxContactPoints: 6000,
    zoomConfig: {
      scale: 1.0,
      min: 0.35,
      max: 3.0,
      step: 0.06,
    },
    baseControl: {
      deadband: 0.12,
      step: 0.012,
      minX: -1.5,
      maxX: 1.5,
      minZ: -2.0,
      maxZ: 1.5,
      minY: 0.1,
      maxY: 2.0,
      heightStep: 0.02,
    },
    inputSendHz: Math.max(1.0, Number(searchParams.get("input_hz") ?? 30.0)),
    defaultDemoArmColors: Object.fromEntries(
      Object.entries(armConfig).map(([armId, arm]) => [armId, arm.color])
    ),
  };
}
