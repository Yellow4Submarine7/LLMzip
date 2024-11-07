import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  // 根据用户语言偏好或URL参数决定使用哪个页面
  const lang = request.cookies.get('lang')?.value || 'en'
  
  if (lang === 'zh') {
    // 加载中文版本
    return NextResponse.rewrite(new URL('/zh', request.url))
  }
  
  // 默认英文版本
  return NextResponse.next()
} 