export function identity3x3() {
  return [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ];
}

export function transpose3x3(rows) {
  return [
    [rows[0][0], rows[1][0], rows[2][0]],
    [rows[0][1], rows[1][1], rows[2][1]],
    [rows[0][2], rows[1][2], rows[2][2]],
  ];
}

export function multiply3x3(a, b) {
  const out = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
  ];
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  return out;
}
