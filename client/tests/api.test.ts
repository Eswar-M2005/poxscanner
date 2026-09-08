import { describe, expect, it } from "vitest";

import { health, modelInfo, predict } from "@/lib/api";

describe("api wrapper exports", () => {
  it("exposes the expected functions", () => {
    expect(typeof health).toBe("function");
    expect(typeof modelInfo).toBe("function");
    expect(typeof predict).toBe("function");
  });
});

