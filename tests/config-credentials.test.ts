import { afterEach, describe, expect, test } from "bun:test";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig } from "../src/config.ts";
import { loadCredentials, saveCredentials } from "../src/credentials.ts";

const temporaryDirectories: string[] = [];

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "kiro-relay-"));
  temporaryDirectories.push(directory);
  return directory;
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true })));
});

describe("configuration and credentials", () => {
  test("loads the minimal YAML configuration", async () => {
    const directory = await temporaryDirectory();
    const configPath = join(directory, "config.yaml");
    await writeFile(
      configPath,
      "listen: 127.0.0.1:4000\nmodels:\n  claude-opus-5:\n    contextLength: 1000000\naccounts:\n  - credentialFile: /tmp/account.json\n",
    );

    const config = await loadConfig(configPath);
    expect(config.listen).toBe("127.0.0.1:4000");
    expect(config.models["claude-opus-5"].contextLength).toBe(1_000_000);
    expect(config.accounts[0].credentialFile).toBe("/tmp/account.json");
  });

  test("atomically writes a complete private credential file", async () => {
    const directory = await temporaryDirectory();
    const credentialPath = join(directory, "account.json");
    const credentials = {
      accessToken: "test-access",
      refreshToken: "test-refresh",
      providerSpecificData: { authMethod: "idc", region: "us-east-1" },
    };

    await saveCredentials(credentialPath, credentials);
    expect(await loadCredentials(credentialPath)).toEqual(credentials);
    expect((await stat(credentialPath)).mode & 0o777).toBe(0o600);
    expect(JSON.parse(await readFile(credentialPath, "utf8"))).toEqual(credentials);
  });
});
