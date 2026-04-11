# VR Prototype Interface

This folder contains the Quest WebXR client and startup scripts for the multi-user soft-arm VR prototype powered by the `virtual_field` package.
Deployment model:
- Linux desktop starts and hosts the VR services.
- Clients join over URLs:
  - Headgear client opens HTTPS URL in Quest Browser.
  - Object-publisher client connects to WSS URL from Python CLI.

## Quick start

Use `npx` or `bunx`, depending on your preference.

### Serve the WebXR client over HTTPS

Serve the WebXR client over HTTPS:

```bash
# In VR directory
npx serve client -l 8443 --ssl-cert certs/dev-cert.pem --ssl-key certs/dev-key.pem
# or
bunx serve client -l 8443 --ssl-cert certs/dev-cert.pem --ssl-key certs/dev-key.pem
```