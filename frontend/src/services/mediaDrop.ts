const MEDIA_EXTENSION = /\.(mp4|mkv|mov|avi|wmv|flv|webm|m4v|mpg|mpeg|ts|mts|m2ts|png|jpe?g|bmp|webp|tiff?|avif|heic|heif|jxl|jxr|wdp|hdp)$/i

export function droppedMediaPath(files: Iterable<File>, resolve: (file: File) => string): string | null {
  for (const file of files) {
    if (!MEDIA_EXTENSION.test(file.name)) continue
    const path = resolve(file)
    if (path) return path
  }
  return null
}
