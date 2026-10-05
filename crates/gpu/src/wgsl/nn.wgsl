// The AI Denoise network's operations (lightcraft-denoise `model::run_cpu`; keep them in step).
// Activations are channel planes, row-major: index (c · h + y) · w + x.

// Convolution, zero padding k / 2, then ReLU. Each invocation computes OCB output channels of
// one pixel (x, y, output-channel block), so every input value it loads feeds OCB sums.
// P: cin, cout, k, stride, relu, w_off, b_off, ih, iw, oh, ow
const OCB: u32 = 8u;

@compute @workgroup_size(16, 16)
fn nn_conv(@builtin(global_invocation_id) g: vec3<u32>) {
    let cin = pu(0u);
    let cout = pu(1u);
    let k = pu(2u);
    let stride = pu(3u);
    let relu = pu(4u);
    let wo = pu(5u);
    let bo = pu(6u);
    let ih = pu(7u);
    let iw = pu(8u);
    let oh = pu(9u);
    let ow = pu(10u);
    let ox = g.x;
    let oy = g.y;
    let oc0 = g.z * OCB;
    if (ox >= ow || oy >= oh || oc0 >= cout) {
        return;
    }
    let n = min(OCB, cout - oc0);
    let kk = k * k;
    let ws = cin * kk; // weights between consecutive output channels
    // two vec4 accumulators (registers, fully unrolled); reads past the last output channel land
    // in other weights or are clamped by robust buffer access, and are never written out
    let bi = bo + oc0;
    var s0 = vec4<f32>(wt[bi], wt[bi + 1u], wt[bi + 2u], wt[bi + 3u]);
    var s1 = vec4<f32>(wt[bi + 4u], wt[bi + 5u], wt[bi + 6u], wt[bi + 7u]);
    let pad = i32(k / 2u);
    for (var ic = 0u; ic < cin; ic++) {
        let xbase = ic * ih * iw;
        let wbase = wo + oc0 * ws + ic * kk;
        for (var ky = 0u; ky < k; ky++) {
            let iy = i32(oy * stride + ky) - pad;
            if (iy < 0 || iy >= i32(ih)) {
                continue;
            }
            for (var kx = 0u; kx < k; kx++) {
                let ix = i32(ox * stride + kx) - pad;
                if (ix < 0 || ix >= i32(iw)) {
                    continue;
                }
                let xv = x[xbase + u32(iy) * iw + u32(ix)];
                let i = wbase + ky * k + kx;
                s0 += vec4<f32>(wt[i], wt[i + ws], wt[i + 2u * ws], wt[i + 3u * ws]) * xv;
                s1 += vec4<f32>(wt[i + 4u * ws], wt[i + 5u * ws], wt[i + 6u * ws], wt[i + 7u * ws]) * xv;
            }
        }
    }
    if (relu != 0u) {
        s0 = max(s0, vec4<f32>(0.0));
        s1 = max(s1, vec4<f32>(0.0));
    }
    let plane = oh * ow;
    let o = (oc0 * oh + oy) * ow + ox;
    for (var j = 0u; j < 4u; j++) {
        if (j < n) {
            y[o + j * plane] = s0[j];
        }
        if (j + 4u < n) {
            y[o + (j + 4u) * plane] = s1[j];
        }
    }
}

// Bilinear ×2 upsampling, pixel-centre aligned, edges clamped. P: c, ih, iw
@compute @workgroup_size(16, 16)
fn nn_up(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = pu(0u);
    let ih = pu(1u);
    let iw = pu(2u);
    let oh = ih * 2u;
    let ow = iw * 2u;
    if (g.x >= ow || g.y >= oh || g.z >= c) {
        return;
    }
    let base = g.z * ih * iw;
    let sy = (f32(g.y) + 0.5) * 0.5 - 0.5;
    let sx = (f32(g.x) + 0.5) * 0.5 - 0.5;
    let y0f = floor(sy);
    let x0f = floor(sx);
    let fy = sy - y0f;
    let fx = sx - x0f;
    let y0 = u32(clamp(i32(y0f), 0, i32(ih) - 1));
    let y1 = u32(clamp(i32(y0f) + 1, 0, i32(ih) - 1));
    let x0 = u32(clamp(i32(x0f), 0, i32(iw) - 1));
    let x1 = u32(clamp(i32(x0f) + 1, 0, i32(iw) - 1));
    let a = x[base + y0 * iw + x0] + (x[base + y0 * iw + x1] - x[base + y0 * iw + x0]) * fx;
    let b = x[base + y1 * iw + x0] + (x[base + y1 * iw + x1] - x[base + y1 * iw + x0]) * fx;
    y[(g.z * oh + g.y) * ow + g.x] = a + (b - a) * fy;
}
