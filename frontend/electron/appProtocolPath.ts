import path from 'node:path'

type PathApi = Pick<typeof path, 'resolve' | 'relative' | 'isAbsolute' | 'sep'>

/** Resolve an app://studio URL to a renderer file, or reject it safely. */
export function resolveAppProtocolPath(requestUrl: string, rendererDist: string, pathApi: PathApi = path): string | null {
  let url: URL
  try {
    url = new URL(requestUrl)
  } catch {
    return null
  }
  if (url.protocol !== 'app:' || url.hostname !== 'studio' || url.username || url.password || url.port) return null

  // URL normalizes dot segments before exposing pathname, so inspect the raw
  // path as well to catch encoded traversal before that normalization occurs.
  const authorityAndPath = requestUrl.match(/^[a-z][a-z\d+.-]*:\/\/[^/?#]*([^?#]*)/i)
  if (!authorityAndPath) return null
  const rawPath = authorityAndPath[1] || '/'
  if (/%2f|%5c/i.test(rawPath)) return null

  let decodedPath: string
  try {
    decodedPath = decodeURIComponent(rawPath)
  } catch {
    return null
  }
  if (decodedPath.includes('\\') || decodedPath.includes('\0')) return null

  const segments = decodedPath.split('/').filter(Boolean)
  if (segments.some(segment => segment === '.' || segment === '..' || segment.includes(':') ||
      /[. ]$/.test(segment) || [...segment].some(character => character.charCodeAt(0) <= 0x1f))) return null
  if (segments.length === 0) segments.push('index.html')

  const root = pathApi.resolve(rendererDist)
  const candidate = pathApi.resolve(root, ...segments)
  const relative = pathApi.relative(root, candidate)
  if (relative === '..' || relative.startsWith(`..${pathApi.sep}`) || pathApi.isAbsolute(relative)) return null
  return candidate
}
