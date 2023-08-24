import { QueryClient, useQuery } from "react-query";
import { ImageRequest } from "./requests";
import { dirName } from "./util/pathUtil";
import { useSnapshot } from "valtio";
import { stateSystem } from "./store";
import { loadOptional } from "./util/loadUtil";

async function getJsonResponse<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`An error occurred: ${res.status}`);
  return await res.json();
}

export function useUsers() {
  return useQuery(["users"], async () => getJsonResponse<string[]>("/api/v1/users"));
}

export function useModels() {
  return useQuery(["models"], async () => getJsonResponse<[string, string, string][]>("/api/v1/models"));
}

export function useBaseModelType(path: string) {
  const queryModels = useModels();

  return useQuery(
    ["baseModelType", path],
    async (): Promise<string | undefined> => {
      return queryModels.data?.find(([_, __, p]) => p === path)?.[1];
    },
    {
      enabled: queryModels.isSuccess && queryModels.data !== null,
    }
  );
}

export function useLoraModels(baseModelType: string | undefined) {
  const queryModels = useModels();

  return useQuery(["loraModels", baseModelType], async () => {
    return queryModels.data?.filter(([t, b, _]) => t == "lora" && b === baseModelType).map(([_, __, p]) => p);
  });
}

export function useControlNetModels(baseModelType: string | undefined) {
  const queryModels = useModels();

  return useQuery(["controlNetModels", baseModelType], async () => {
    return queryModels.data?.filter(([t, b, _]) => t == "controlnet" && b === baseModelType).map(([_, __, p]) => p);
  });
}

export function useCollections() {
  const snapSystem = useSnapshot(stateSystem);
  const user = snapSystem.user;

  return useQuery(["collections", user], async () => getJsonResponse<string[]>(`/api/v1/collections/${user}`));
}

export function useImages(collection: string | null) {
  const snapSystem = useSnapshot(stateSystem);
  const user = snapSystem.user;

  return useQuery(
    ["images", user, collection],
    async () => getJsonResponse<string[]>(`/api/v1/images/${user}/${collection}`),
    {
      enabled: collection !== null,
    }
  );
}

export function useImageInfo(path: string | null) {
  const snapSystem = useSnapshot(stateSystem);
  const user = snapSystem.user;

  return useQuery(["image-info", user, path], async () => getJsonResponse<any>(`/api/v1/metadata/${user}/${path}`), {
    enabled: path !== null,
  });
}

export function useMetadata(path: string | null) {
  const snapSystem = useSnapshot(stateSystem);
  const user = snapSystem.user;

  const imageInfo = useImageInfo(path);

  return useQuery(
    ["metadata", user, path],
    async (): Promise<ImageRequest | null> => {
      const sd_data = imageInfo.data["seed-alchemy"];
      return loadOptional(JSON.parse(sd_data ?? "{}"), ImageRequest);
    },
    {
      enabled: imageInfo.isSuccess && imageInfo.data != null,
    }
  );
}

export function addImages(queryClient: QueryClient, user: string, imagePaths: string[]) {
  const groupedByDir = imagePaths.reduce((groups, path) => {
    const dir = dirName(path);
    if (!groups[dir]) {
      groups[dir] = [];
    }
    groups[dir].push(path);
    return groups;
  }, {} as Record<string, string[]>);

  for (const collection in groupedByDir) {
    const filesInDir = groupedByDir[collection];

    queryClient.setQueryData<string[] | undefined>(["images", user, collection], (oldImages) => {
      if (oldImages === undefined) return undefined;
      return [...filesInDir.reverse(), ...oldImages];
    });
  }
}

export function removeImage(queryClient: QueryClient, user: string, imagePath: string) {
  const collection = dirName(imagePath);

  queryClient.setQueryData<string[] | undefined>(["images", user, collection], (oldImages) => {
    if (oldImages === undefined) return undefined;
    return oldImages.filter((item) => item != imagePath);
  });
}
