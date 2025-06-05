/*
 * Copyright 2016 - 2024 Drew Edwards, tmpim
 *
 * This file is part of Krist.
 *
 * Krist is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Krist is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Krist. If not, see <http://www.gnu.org/licenses/>.
 *
 * For more project information, see <https://github.com/tmpim/krist>.
 */

import { Address, Limit, Offset, PaginatedResult } from "../database/index.js";
import { ErrorAddressNotFound, ErrorInvalidParameter, ErrorMissingParameter } from "../errors/index.js";
import { getAddress, getAddresses, getRichAddresses } from "../krist/addresses/index.js";
import { isValidBcknAddress, makeV2Address, validateLimitOffset } from "../utils/index.js";

export async function ctrlGetAddresses(
  limit: Limit,
  offset: Offset
): Promise<PaginatedResult<Address>> {
  await validateLimitOffset(limit, offset);
  return getAddresses(limit, offset);
}

export async function ctrlGetRichAddresses(
  limit: Limit,
  offset: Offset
): Promise<PaginatedResult<Address>> {
  await validateLimitOffset(limit, offset);
  return getRichAddresses(limit, offset);
}

export async function ctrlGetAddress(
  address?: string,
  fetchNames?: boolean
): Promise<Address> {
  if (!address) throw new ErrorMissingParameter("address");
  if (!isValidBcknAddress(address)) throw new ErrorInvalidParameter("address");

  const result = await getAddress(address, !!fetchNames);
  if (!result) throw new ErrorAddressNotFound(address);

  return result;
}

export async function ctrlGetAddressAlert(
  privatekey?: string
): Promise<string | null> {
  if (!privatekey) throw new ErrorMissingParameter("privatekey");
  const address = makeV2Address(privatekey);
  const result = await getAddress(address);
  if (!result) throw new ErrorAddressNotFound(address);
  return result.alert ?? null;
}
