/* Portable scan exports. Coordinates supplied by the viewer are feet, Y up. */
const DRUScanExports = (() => {
  const encoder = new TextEncoder();

  function crc32(bytes) {
    let crc = 0xffffffff;
    for (const byte of bytes) {
      crc ^= byte;
      for (let bit = 0; bit < 8; bit++) {
        crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
      }
    }
    return (crc ^ 0xffffffff) >>> 0;
  }

  function zip(entries) {
    // Align payloads to 64 bytes. USDZ uses stored entries, matching the iOS packager.
    if (entries.length > 65535) throw new Error("This export contains too many ZIP entries.");
    const local = [], directory = [];
    let offset = 0, directorySize = 0;
    for (const entry of entries) {
      const name = encoder.encode(entry.name);
      const raw = typeof entry.data === "string" ? encoder.encode(entry.data) : entry.data;
      const data = entry.compressed || raw;
      const method = entry.compressed ? 8 : 0;
      let padding = (64 - ((offset + 30 + name.length) % 64)) % 64;
      if (padding > 0 && padding < 4) padding += 64;
      if (raw.length > 0xffffffff || data.length > 0xffffffff || offset + 30 + name.length + padding + data.length > 0xffffffff) {
        throw new Error("This export exceeds the supported 4 GB ZIP size.");
      }
      const checksum = crc32(raw);
      const header = new Uint8Array(30 + name.length + padding);
      const view = new DataView(header.buffer);
      view.setUint32(0, 0x04034b50, true);
      view.setUint16(4, 20, true);
      view.setUint16(8, method, true);
      view.setUint32(14, checksum, true);
      view.setUint32(18, data.length, true);
      view.setUint32(22, raw.length, true);
      view.setUint16(26, name.length, true);
      view.setUint16(28, padding, true);
      header.set(name, 30);
      if (padding) view.setUint16(32 + name.length, padding - 4, true);
      local.push(header, data);
      const central = new Uint8Array(46 + name.length);
      const index = new DataView(central.buffer);
      index.setUint32(0, 0x02014b50, true);
      index.setUint16(4, 20, true);
      index.setUint16(6, 20, true);
      index.setUint16(10, method, true);
      index.setUint32(16, checksum, true);
      index.setUint32(20, data.length, true);
      index.setUint32(24, raw.length, true);
      index.setUint16(28, name.length, true);
      index.setUint32(42, offset, true);
      central.set(name, 46);
      directory.push(central);
      directorySize += central.length;
      offset += header.length + data.length;
    }
    const end = new Uint8Array(22);
    const footer = new DataView(end.buffer);
    footer.setUint32(0, 0x06054b50, true);
    footer.setUint16(8, entries.length, true);
    footer.setUint16(10, entries.length, true);
    footer.setUint32(12, directorySize, true);
    footer.setUint32(16, offset, true);
    const archive = new Uint8Array(offset + directorySize + end.length);
    let cursor = 0;
    for (const part of [...local, ...directory, end]) {
      archive.set(part, cursor);
      cursor += part.length;
    }
    return archive;
  }

  async function compressedZip(entries) {
    if (typeof CompressionStream === "undefined") return zip(entries);
    let compressor;
    try {
      compressor = new CompressionStream("deflate-raw");
    } catch {
      return zip(entries);
    }
    const packed = [];
    for (const entry of entries) {
      const raw = typeof entry.data === "string" ? encoder.encode(entry.data) : entry.data;
      const stream = new Blob([raw]).stream().pipeThrough(compressor);
      const compressed = new Uint8Array(await new Response(stream).arrayBuffer());
      packed.push({name: entry.name, data: raw, compressed: compressed.length < raw.length ? compressed : null});
      compressor = new CompressionStream("deflate-raw");
    }
    return zip(packed);
  }

  function obj(positions, colors, indices, markers, notice) {
    const lines = [`# ${notice}`, "# Units feet, local Y-up frame. No survey reference."];
    for (let i = 0; i < positions.length; i += 3) {
      const point = Array.from(positions.subarray(i, i + 3), value => value.toFixed(6));
      const color = Array.from(colors.subarray(i, i + 3), value => (value / 255).toFixed(6));
      lines.push(`v ${point.join(" ")} ${color.join(" ")}`);
    }
    for (let i = 0; i < indices.length; i += 3) {
      lines.push(`f ${indices[i] + 1} ${indices[i + 1] + 1} ${indices[i + 2] + 1}`);
    }
    let next = positions.length / 3 + 1;
    for (const marker of markers) {
      lines.push("# Recorded field measurement, feet");
      lines.push(`v ${marker.slice(0, 3).join(" ")}`, `v ${marker.slice(3, 6).join(" ")}`);
      lines.push(`l ${next} ${next + 1}`);
      next += 2;
    }
    return lines.join("\n") + "\n";
  }

  function pointCloud(positions, colors, notice) {
    const lines = ["ply", "format ascii 1.0", `comment ${notice}`,
      "comment Reconstructed model vertices, metres, local Y-up frame. No survey reference.",
      `element vertex ${positions.length / 3}`, "property float x", "property float y", "property float z",
      "property uchar red", "property uchar green", "property uchar blue", "end_header"];
    for (let i = 0; i < positions.length; i += 3) {
      const point = Array.from(positions.subarray(i, i + 3), value => (value * 0.3048).toFixed(6));
      lines.push(`${point.join(" ")} ${colors[i]} ${colors[i + 1]} ${colors[i + 2]}`);
    }
    return lines.join("\n") + "\n";
  }

  function floatCloud(positions) {
    // Same packed little-endian xyz float32 layout as iOS ScanCloudStore, in metres.
    const bytes = new Uint8Array(positions.length * 4);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < positions.length; i++) view.setFloat32(i * 4, positions[i] * 0.3048, true);
    return bytes;
  }

  function usdz(positions, colors, indices, markers, notice) {
    const points = [], displayColors = [];
    for (let i = 0; i < positions.length; i += 3) {
      points.push(`(${Array.from(positions.subarray(i, i + 3), value => value.toFixed(6)).join(", ")})`);
      const linear = Array.from(colors.subarray(i, i + 3), value => {
        const srgb = value / 255;
        return (srgb <= 0.04045 ? srgb / 12.92 : ((srgb + 0.055) / 1.055) ** 2.4).toFixed(6);
      });
      displayColors.push(`(${linear.join(", ")})`);
    }
    const layer = ["#usda 1.0", "(", '    defaultPrim = "DRUScan"', "    metersPerUnit = 0.3048",
      '    upAxis = "Y"', `    customLayerData = { string notice = ${JSON.stringify(notice)} }`, ")",
      'def Xform "DRUScan"', "{", '    def Mesh "Model"', "    {",
      '        uniform token subdivisionScheme = "none"', "        uniform bool doubleSided = true",
      `        point3f[] points = [${points.join(", ")}]`,
      `        int[] faceVertexCounts = [${Array(indices.length / 3).fill(3).join(", ")}]`,
      `        int[] faceVertexIndices = [${Array.from(indices).join(", ")}]`,
      `        color3f[] primvars:displayColor = [${displayColors.join(", ")}] (interpolation = "vertex")`, "    }"];
    for (let i = 0; i < markers.length; i++) {
      const marker = markers[i];
      layer.push(`    def BasisCurves "FieldMeasurement${i + 1}"`, "    {",
        '        uniform token type = "linear"', "        int[] curveVertexCounts = [2]",
        `        point3f[] points = [(${marker.slice(0, 3).join(", ")}), (${marker.slice(3, 6).join(", ")})]`,
        "        float[] widths = [0.05]", "        color3f[] primvars:displayColor = [(1, 0.15, 0.15)]", "    }");
    }
    layer.push("}");
    return zip([{name: "scan.usda", data: layer.join("\n") + "\n"}]);
  }

  return {zip, compressedZip, obj, pointCloud, floatCloud, usdz};
})();
